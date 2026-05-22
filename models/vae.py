"""
Variational Autoencoder (VAE) trained on CIFAR-10.

Architecture:
  Encoder: 4× Conv2d(k=4, s=2, p=1) → flatten → FC heads (mu, logvar)
  Decoder: FC → view → 4× ConvTranspose2d → Tanh
  Latent dim: 256, beta-annealed KL weight over 10 warmup epochs
"""

import os, json, csv, time, math, random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.utils as vutils


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CFG:
    data_root: str = "./data"
    out_dir: str = "./outputs/vae"
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    z_dim: int = 256
    num_workers: int = 2
    log_interval: int = 100
    beta_warmup_epochs: int = 10
    beta_start: float = 0.0
    beta_end: float = 1.0
    grid_n: int = 64
    samples_export_total: int = 10_000
    samples_export_bs: int = 100
    smoke_subset: int = 0
    use_amp: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


def save_grid_pair(orig, recon, path, nrow=8):
    grid_o = vutils.make_grid(denorm(orig), nrow=nrow)
    grid_r = vutils.make_grid(denorm(recon), nrow=nrow)
    vutils.save_image(torch.cat([grid_o, grid_r], dim=1), path)


def beta_schedule(epoch, warmup_epochs=10, beta_start=0.0, beta_end=1.0):
    if epoch <= warmup_epochs:
        return beta_start + (epoch / max(1, warmup_epochs)) * (beta_end - beta_start)
    return beta_end


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """(B,3,32,32) → mu, logvar each (B, z_dim)"""

    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1, bias=False), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,  64,  4, 2, 1, bias=False), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc_mu     = nn.Linear(256 * 2 * 2, z_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, z_dim)

    def forward(self, x):
        h = self.flatten(self.net(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """z (B, z_dim) → (B, 3, 32, 32) in [-1, 1]"""

    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 2 * 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,   32, 4, 2, 1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,    3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 2, 2)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z  = reparameterize(mu, logvar)
        xr = self.dec(z)
        return xr, mu, logvar, z


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(x, xr, mu, logvar, beta=1.0):
    recon = F.mse_loss(xr, x, reduction="mean")
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(cfg: CFG):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = datasets.CIFAR10(cfg.data_root, train=True,  download=True, transform=tfm)
    test  = datasets.CIFAR10(cfg.data_root, train=False, download=True, transform=tfm)
    if cfg.smoke_subset > 0:
        train = Subset(train, list(range(cfg.smoke_subset)))
        test  = Subset(test,  list(range(min(len(test), cfg.smoke_subset // 5))))
    kw = dict(num_workers=cfg.num_workers, pin_memory=True)
    return (DataLoader(train, batch_size=cfg.batch_size, shuffle=True,  **kw),
            DataLoader(test,  batch_size=cfg.batch_size, shuffle=False, **kw))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: CFG | None = None):
    if cfg is None:
        cfg = CFG()
    set_seed(42)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    trainloader, testloader = get_loaders(cfg)
    vae = VAE(z_dim=cfg.z_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))

    vae.eval()
    with torch.no_grad():
        fixed_imgs, _ = next(iter(testloader))
    fixed_imgs = fixed_imgs[:cfg.grid_n].to(device)
    vae.train()

    csv_path = os.path.join(cfg.out_dir, "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "beta", "avg_total", "avg_recon", "avg_kl", "time_sec"])

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        vae.train()
        ep_total = ep_recon = ep_kl = 0.0
        beta = beta_schedule(epoch, cfg.beta_warmup_epochs, cfg.beta_start, cfg.beta_end)

        for i, (x, _) in enumerate(trainloader, 1):
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device == "cuda")):
                xr, mu, logvar, _ = vae(x)
                total, recon, kl = vae_loss(x, xr, mu, logvar, beta)
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            ep_total += total.item(); ep_recon += recon.item(); ep_kl += kl.item()

        n = len(trainloader); dt = time.time() - t0
        avg_total, avg_recon, avg_kl = ep_total / n, ep_recon / n, ep_kl / n
        print(f"Epoch {epoch:03d} | β={beta:.3f} | {dt:.1f}s | total={avg_total:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f}")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, beta, avg_total, avg_recon, avg_kl, round(dt, 2)])

        vae.eval()
        with torch.no_grad():
            xr_fixed, _, _, _ = vae(fixed_imgs)
            save_grid_pair(fixed_imgs, xr_fixed, os.path.join(cfg.out_dir, f"recon_epoch_{epoch:03d}.png"))
            z_rand = torch.randn(cfg.grid_n, cfg.z_dim, device=device)
            xs = vae.dec(z_rand)
        vutils.save_image(vutils.make_grid(denorm(xs), nrow=8),
                          os.path.join(cfg.out_dir, f"samples_epoch_{epoch:03d}.png"))
        torch.save({"epoch": epoch, "model": vae.state_dict(),
                    "optimizer": opt.state_dict(), "cfg": cfg.__dict__},
                   os.path.join(cfg.out_dir, f"vae_epoch_{epoch:03d}.pth"))

    print("Training complete. Outputs in:", cfg.out_dir)
    return vae


# ---------------------------------------------------------------------------
# Sample generation from a saved checkpoint
# ---------------------------------------------------------------------------

def generate(ckpt_path: str, n: int = 64, out_path: str = "vae_samples.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    z_dim = ckpt["cfg"]["z_dim"]
    vae = VAE(z_dim=z_dim).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    with torch.no_grad():
        z  = torch.randn(n, z_dim, device=device)
        xs = vae.dec(z)
    vutils.save_image(vutils.make_grid(denorm(xs), nrow=8), out_path)
    print("Saved:", out_path)
    return xs


if __name__ == "__main__":
    train()
