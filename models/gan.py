"""
Deep Convolutional GAN (DCGAN) trained on CIFAR-10.

Architecture:
  Generator:     z(128,1,1) → ConvTranspose2d ×5 → (3,32,32) Tanh
  Discriminator: (3,32,32) → Conv2d ×4 → logit scalar
  Loss: BCEWithLogitsLoss + AMP mixed precision
"""

import os, random, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """z (N, latent_dim, 1, 1) → (N, 3, 32, 32)"""

    def __init__(self, latent_dim: int = 128, fm: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fm * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(fm * 8), nn.ReLU(True),
            nn.ConvTranspose2d(fm * 8,    fm * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(fm * 4), nn.ReLU(True),
            nn.ConvTranspose2d(fm * 4,    fm * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(fm * 2), nn.ReLU(True),
            nn.ConvTranspose2d(fm * 2,    fm,     4, 2, 1, bias=False), nn.BatchNorm2d(fm),     nn.ReLU(True),
            nn.ConvTranspose2d(fm,         3,     3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """(N, 3, 32, 32) → (N,) logits"""

    def __init__(self, fm: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,      fm,     3, 1, 1, bias=False),                          nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fm,     fm * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(fm * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fm * 2, fm * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(fm * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fm * 4, fm * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(fm * 8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Conv2d(fm * 8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        return self.head(self.features(x)).view(-1)


def weights_init(m):
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loader(data_root: str = "./data", batch_size: int = 128, num_workers: int = 2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    out_dir: str = "./outputs/gan",
    data_root: str = "./data",
    epochs: int = 100,
    latent_dim: int = 128,
    gen_fm: int = 128,
    disc_fm: int = 64,
    batch_size: int = 128,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
    num_workers: int = 2,
):
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    config = dict(epochs=epochs, latent_dim=latent_dim, gen_fm=gen_fm, disc_fm=disc_fm,
                  batch_size=batch_size, lr_g=lr_g, lr_d=lr_d)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    trainloader = get_loader(data_root, batch_size, num_workers)

    G = Generator(latent_dim, gen_fm).to(device)
    D = Discriminator(disc_fm).to(device)
    G.apply(weights_init); D.apply(weights_init)

    optG = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2))
    optD = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        for real, _ in trainloader:
            real = real.to(device)
            N    = real.size(0)
            real_lbl = torch.ones(N, device=device)
            fake_lbl = torch.zeros(N, device=device)

            # Train D
            optD.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss_real = criterion(D(real), real_lbl)
                z         = torch.randn(N, latent_dim, 1, 1, device=device)
                loss_fake = criterion(D(G(z).detach()), fake_lbl)
                loss_D    = loss_real + loss_fake
            scaler.scale(loss_D).backward(); scaler.step(optD)

            # Train G
            optG.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                z      = torch.randn(N, latent_dim, 1, 1, device=device)
                loss_G = criterion(D(G(z)), real_lbl)
            scaler.scale(loss_G).backward(); scaler.step(optG)
            scaler.update()

        G.eval(); D.eval()
        with torch.no_grad():
            fake_fixed = G(fixed_noise)
        grid_real = make_grid(denorm(real[:64].cpu()), nrow=8)
        grid_fake = make_grid(denorm(fake_fixed.cpu()), nrow=8)
        save_image(torch.cat([grid_real, grid_fake], dim=1),
                   os.path.join(out_dir, f"real_vs_fake_epoch_{epoch:03d}.png"))
        torch.save({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                    "LATENT_DIM": latent_dim, "GEN_FM": gen_fm, "DISC_FM": disc_fm},
                   os.path.join(out_dir, f"dcgan_epoch_{epoch:03d}.pth"))
        print(f"Epoch {epoch:03d} | loss_D={loss_D.item():.4f} loss_G={loss_G.item():.4f}")

    print("Training complete. Files in:", out_dir)
    return G, D


# ---------------------------------------------------------------------------
# Sample generation from a saved checkpoint
# ---------------------------------------------------------------------------

def generate(ckpt_path: str, n: int = 64, out_path: str = "gan_samples.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    latent_dim = ckpt.get("LATENT_DIM", 128)
    gen_fm     = ckpt.get("GEN_FM", 128)
    G = Generator(latent_dim, gen_fm).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    with torch.no_grad():
        z    = torch.randn(n, latent_dim, 1, 1, device=device)
        imgs = denorm(G(z)).cpu()
    save_image(make_grid(imgs, nrow=8), out_path)
    print("Saved:", out_path)
    return imgs


# ---------------------------------------------------------------------------
# Export 10k samples for FID/IS evaluation
# ---------------------------------------------------------------------------

def export_samples(G: Generator, latent_dim: int, out_dir: str, total: int = 10_000, bs: int = 256):
    device = next(G.parameters()).device
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    G.eval()
    saved = 0
    with torch.no_grad():
        while saved < total:
            cur  = min(bs, total - saved)
            z    = torch.randn(cur, latent_dim, 1, 1, device=device)
            imgs = denorm(G(z)).cpu()
            for i in range(cur):
                save_image(imgs[i], os.path.join(out_dir, f"{saved + i:05d}.png"))
            saved += cur
    print(f"Saved {saved} images to {out_dir}")


if __name__ == "__main__":
    G, D = train()
