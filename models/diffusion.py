"""
Improved Denoising Diffusion Probabilistic Model (DDPM) trained on CIFAR-10.

Architecture:
  UNet with sinusoidal time embeddings, residual blocks, multi-head self-attention,
  group normalization, and EMA-smoothed weights.
  Cosine noise schedule over T=1000 steps.
"""

import os, random, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


# ---------------------------------------------------------------------------
# UNet components
# ---------------------------------------------------------------------------

def _valid_groups(channels: int, preferred: int = 8) -> int:
    g = preferred
    while channels % g != 0 and g > 1:
        g -= 1
    return max(1, g)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        factor = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -factor)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_valid_groups(channels), channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        nh, hd = self.num_heads, C // self.num_heads
        q = q.view(B, nh, hd, H * W)
        k = k.view(B, nh, hd, H * W)
        v = v.view(B, nh, hd, H * W)
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q, k) * hd ** -0.5, dim=-1)
        out  = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.block1 = nn.Sequential(
            nn.GroupNorm(_valid_groups(in_ch), in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(_valid_groups(out_ch), out_ch), nn.SiLU(),
            nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x) + self.time_mlp(t_emb)[:, :, None, None]
        return self.block2(h) + self.res_conv(x)


class UNet(nn.Module):
    """
    32×32 UNet: 32→16→8 (bottleneck) →16→32
    Channels: base=128, 2×=256, 4×=512
    """

    def __init__(self, img_ch: int = 3, base_channels: int = 128,
                 time_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        ch1, ch2, ch3 = base_channels, base_channels * 2, base_channels * 4
        self.down1 = nn.ModuleList([ResBlock(img_ch, ch1, time_dim, dropout),
                                    ResBlock(ch1,    ch1, time_dim, dropout)])
        self.down2 = nn.ModuleList([ResBlock(ch1, ch2, time_dim, dropout),
                                    ResBlock(ch2, ch2, time_dim, dropout)])
        self.attn2 = SelfAttention(ch2)
        self.down3 = nn.ModuleList([ResBlock(ch2, ch3, time_dim, dropout),
                                    ResBlock(ch3, ch3, time_dim, dropout)])
        self.attn3 = SelfAttention(ch3)
        self.pool  = nn.AvgPool2d(2)

        self.mid1     = ResBlock(ch3, ch3, time_dim, dropout)
        self.mid_attn = SelfAttention(ch3)
        self.mid2     = ResBlock(ch3, ch3, time_dim, dropout)

        self.up2      = nn.ModuleList([ResBlock(ch3 + ch2, ch2, time_dim, dropout),
                                       ResBlock(ch2,       ch2, time_dim, dropout)])
        self.attn_up2 = SelfAttention(ch2)
        self.up1      = nn.ModuleList([ResBlock(ch2 + ch1, ch1, time_dim, dropout),
                                       ResBlock(ch1,       ch1, time_dim, dropout)])
        self.attn_up1 = SelfAttention(ch1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.final    = nn.Sequential(
            nn.GroupNorm(_valid_groups(ch1), ch1), nn.SiLU(),
            nn.Conv2d(ch1, img_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = x
        for blk in self.down1: h = blk(h, t_emb)
        d1 = h
        h = self.pool(h)
        for blk in self.down2: h = blk(h, t_emb)
        h = self.attn2(h); d2 = h
        h = self.pool(h)
        for blk in self.down3: h = blk(h, t_emb)
        h = self.attn3(h)
        h = self.mid1(h, t_emb); h = self.mid_attn(h); h = self.mid2(h, t_emb)
        h = self.upsample(h)
        h = torch.cat([h, d2], dim=1)
        for blk in self.up2: h = blk(h, t_emb)
        h = self.attn_up2(h)
        h = self.upsample(h)
        h = torch.cat([h, d1], dim=1)
        for blk in self.up1: h = blk(h, t_emb)
        h = self.attn_up1(h)
        return self.final(h)


# ---------------------------------------------------------------------------
# Diffusion schedule (cosine, T=1000)
# ---------------------------------------------------------------------------

T = 1000

def _cosine_betas(timesteps: int = T, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    return torch.clip(1 - ac[1:] / ac[:-1], 0.0001, 0.9999)


class DiffusionSchedule:
    """Pre-computes all diffusion schedule tensors and moves them to device."""

    def __init__(self, timesteps: int = T, device: str = "cpu"):
        betas = _cosine_betas(timesteps)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        ac_prev = torch.cat([torch.tensor([1.0]), ac[:-1]])
        self.T        = timesteps
        self.betas    = betas.to(device)
        self.alphas   = alphas.to(device)
        self.ac       = ac.to(device)
        self.ac_prev  = ac_prev.to(device)
        self.sqrt_ac              = torch.sqrt(ac).to(device)
        self.sqrt_one_minus_ac    = torch.sqrt(1.0 - ac).to(device)
        self.sqrt_recip_alphas    = torch.sqrt(1.0 / alphas).to(device)
        self.posterior_var        = (betas * (1.0 - ac_prev) / (1.0 - ac)).to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sa  = self.sqrt_ac[t][:, None, None, None]
        sm  = self.sqrt_one_minus_ac[t][:, None, None, None]
        return sa * x0 + sm * noise

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        t     = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t   = self.q_sample(x0, t, noise)
        return F.mse_loss(model(x_t, t), noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t_idx: int) -> torch.Tensor:
        b = x.size(0)
        t = torch.full((b,), t_idx, device=x.device, dtype=torch.long)
        eps    = model(x, t)
        beta_t = self.betas[t_idx]
        mean   = self.sqrt_recip_alphas[t_idx] * (
            x - beta_t / self.sqrt_one_minus_ac[t_idx] * eps
        )
        if t_idx == 0:
            return mean
        return mean + torch.sqrt(self.posterior_var[t_idx]) * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple) -> torch.Tensor:
        model.eval()
        img = torch.randn(shape, device=self.betas.device)
        for t in reversed(range(self.T)):
            img = self.p_sample(model, img, t)
        return img


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup: dict = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loader(data_root: str = "./data", batch_size: int = 64, num_workers: int = 2):
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    out_dir: str = "./outputs/diffusion",
    data_root: str = "./data",
    epochs: int = 500,
    batch_size: int = 64,
    lr: float = 2e-4,
    base_channels: int = 128,
    time_dim: int = 256,
    dropout: float = 0.1,
    ema_decay: float = 0.9999,
    save_every: int = 50,
    num_workers: int = 2,
):
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    config = dict(epochs=epochs, batch_size=batch_size, lr=lr,
                  base_channels=base_channels, time_dim=time_dim)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    schedule   = DiffusionSchedule(T, device=device)
    trainloader = get_loader(data_root, batch_size, num_workers)
    model      = UNet(img_ch=3, base_channels=base_channels,
                      time_dim=time_dim, dropout=dropout).to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    ema        = EMA(model, decay=ema_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        last_batch = None
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{epochs}")
        for x0, _ in pbar:
            x0 = x0.to(device)
            loss = schedule.loss(model, x0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update()
            epoch_loss += loss.item()
            last_batch = x0
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        print(f"Epoch {epoch} | avg_loss={epoch_loss / len(trainloader):.4f}")

        if epoch % save_every == 0 or epoch == epochs:
            ema.apply_shadow()
            with torch.no_grad():
                samples = denorm(schedule.sample(model, (64, 3, 32, 32))).cpu()
                reals   = denorm(last_batch[:64].cpu())
            save_image(torch.cat([make_grid(reals, nrow=8), make_grid(samples, nrow=8)], dim=1),
                       os.path.join(out_dir, f"real_vs_diffusion_epoch_{epoch:03d}.png"))
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "ema": ema.shadow, "optimizer": optimizer.state_dict(), "T": T},
                       os.path.join(out_dir, f"diffusion_epoch_{epoch:03d}.pth"))
            ema.restore()

    print("Training complete. Outputs in:", out_dir)
    return model, ema


# ---------------------------------------------------------------------------
# Load checkpoint + generate
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str, device: str) -> nn.Module:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet(img_ch=3, base_channels=128, time_dim=256, dropout=0.1).to(device)
    ema_state = ckpt.get("ema", ckpt["model"])
    if isinstance(ema_state, dict) and all(k in model.state_dict() for k in list(ema_state)[:5]):
        model.load_state_dict(ema_state)
    else:
        ema_obj = EMA(model); ema_obj.shadow = ema_state; ema_obj.apply_shadow()
    model.eval()
    return model


def generate(ckpt_path: str, n: int = 64, out_path: str = "diffusion_samples.png"):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model    = _load_model(ckpt_path, device)
    schedule = DiffusionSchedule(T, device=device)
    samples  = denorm(schedule.sample(model, (n, 3, 32, 32))).cpu()
    save_image(make_grid(samples, nrow=8), out_path)
    print("Saved:", out_path)
    return samples


def export_samples(ckpt_path: str, out_dir: str, total: int = 10_000, bs: int = 100):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model    = _load_model(ckpt_path, device)
    schedule = DiffusionSchedule(T, device=device)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    while saved < total:
        cur     = min(bs, total - saved)
        samples = denorm(schedule.sample(model, (cur, 3, 32, 32))).cpu()
        for i in range(cur):
            save_image(samples[i], os.path.join(out_dir, f"{saved + i:05d}.png"))
        saved += cur
        print(f"  {saved}/{total}", end="\r")
    print(f"\nSaved {saved} diffusion samples to {out_dir}")


if __name__ == "__main__":
    train()
