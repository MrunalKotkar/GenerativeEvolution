"""
Streamlit demo — Image Generation Comparison: VAE vs GAN vs Diffusion

Run:
    streamlit run app.py
"""

import os
import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Image Generation Comparison",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path("samples")
ASSETS_DIR  = Path("assets")
EVAL_DIR    = ASSETS_DIR / "evaluation_graphs"

MODEL_LABELS = {
    "vae":       "VAE (Variational Autoencoder)",
    "gan":       "GAN (DCGAN)",
    "diffusion": "Diffusion Model (DDPM)",
}

METRICS = {
    "Model":          ["VAE",      "GAN (DCGAN)", "Diffusion"],
    "Epochs":         [100,        100,            500],
    "Best FID ↓":     [242.15,     31.93,          13.48],
    "Best IS ↑":      [2.06,       6.94,           8.31],
    "Training time":  ["~2 h",     "~1.5 h",       "~18 h"],
    "Latent dim":     [256,        128,            "N/A (T=1000)"],
}

ARCH_IMAGES = {
    "VAE":       ASSETS_DIR / "VAE Architecture.png",
    "GAN":       ASSETS_DIR / "GAN Architecture.png",
    "Diffusion": ASSETS_DIR / "Diffusion Model Architecture.png",
}

EVAL_PLOTS = {
    "VAE":       EVAL_DIR / "vae_metrics_vs_epochs.png",
    "GAN":       EVAL_DIR / "gan_metrics_vs_epochs.png",
    "Diffusion": EVAL_DIR / "diffusion_metrics_vs_epochs.png",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_sample_images(model: str, n: int = 25) -> list[Image.Image]:
    folder = SAMPLES_DIR / model
    files  = sorted(glob.glob(str(folder / "*.png")))
    if not files:
        return []
    chosen = random.sample(files, min(n, len(files)))
    return [Image.open(f).convert("RGB") for f in chosen]


@st.cache_data
def load_metrics_csv(model: str) -> pd.DataFrame | None:
    path = EVAL_DIR / f"{model}_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def image_grid(images: list[Image.Image], cols: int = 5):
    rows = [images[i:i + cols] for i in range(0, len(images), cols)]
    for row in rows:
        cols_st = st.columns(len(row))
        for col, img in zip(cols_st, row):
            col.image(img, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Sample Gallery", "Metrics & Training", "Architecture", "About"],
)

# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Generative Model Comparison on CIFAR-10")
    st.markdown(
        """
        This project trains and evaluates three families of generative models on CIFAR-10
        (32×32 colour images, 10 classes):

        | Model | Core idea |
        |-------|-----------|
        | **VAE** | Encoder → Gaussian latent → Decoder; optimises ELBO (reconstruction + KL) |
        | **GAN** | Generator fools a Discriminator trained adversarially |
        | **Diffusion** | Learns to reverse a Markovian noising process over T=1 000 steps |

        Key metrics (lower FID = better; higher IS = better):
        """
    )

    df = pd.DataFrame(METRICS)
    st.dataframe(df.set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.subheader("Best-epoch generated samples")
    cols = st.columns(3)
    for col, (key, label) in zip(cols, MODEL_LABELS.items()):
        imgs = load_sample_images(key, n=9)
        col.markdown(f"**{label}**")
        if imgs:
            grid = [imgs[i : i + 3] for i in range(0, 9, 3)]
            for row in grid:
                subcols = col.columns(3)
                for sc, img in zip(subcols, row):
                    sc.image(img, use_container_width=True)
        else:
            col.info(f"No pre-generated samples found in `samples/{key}/`.")

# ---------------------------------------------------------------------------
# Sample Gallery page
# ---------------------------------------------------------------------------

elif page == "Sample Gallery":
    st.title("Sample Gallery")
    model_key = st.selectbox(
        "Select model",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda k: MODEL_LABELS[k],
    )
    n_show = st.slider("Images to show", min_value=5, max_value=50, value=25, step=5)

    images = load_sample_images(model_key, n=n_show)
    if images:
        st.markdown(f"Showing **{len(images)}** random samples from `samples/{model_key}/`")
        image_grid(images, cols=5)
    else:
        st.warning(
            f"No images found in `samples/{model_key}/`. "
            "Run the corresponding training script and save ~50 images there."
        )

# ---------------------------------------------------------------------------
# Metrics & Training page
# ---------------------------------------------------------------------------

elif page == "Metrics & Training":
    st.title("Metrics & Training Curves")

    st.subheader("FID / IS summary at best checkpoint")
    df = pd.DataFrame(METRICS).set_index("Model")
    st.dataframe(df, use_container_width=True)

    st.markdown(
        """
        **FID** (Fréchet Inception Distance) measures the distance between the distribution of
        generated images and real CIFAR-10 images in Inception feature space — lower is better.

        **IS** (Inception Score) measures both quality and diversity of generated images — higher is better.
        """
    )

    st.markdown("---")
    st.subheader("Training curves (FID & IS vs epochs)")

    tab_vae, tab_gan, tab_diff = st.tabs(["VAE", "GAN", "Diffusion"])

    for tab, model_name, model_key in [
        (tab_vae,  "VAE",       "vae"),
        (tab_gan,  "GAN",       "gan"),
        (tab_diff, "Diffusion", "diffusion"),
    ]:
        with tab:
            plot_path = EVAL_PLOTS[model_name]
            if plot_path.exists():
                st.image(str(plot_path), use_container_width=True)
            else:
                st.info(f"Plot not found at `{plot_path}`.")

            df_csv = load_metrics_csv(model_key)
            if df_csv is not None:
                st.dataframe(df_csv, use_container_width=True)

# ---------------------------------------------------------------------------
# Architecture page
# ---------------------------------------------------------------------------

elif page == "Architecture":
    st.title("Model Architectures")

    for model_name, img_path in ARCH_IMAGES.items():
        st.subheader(f"{model_name} Architecture")
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info(f"Diagram not found at `{img_path}`.")
        st.markdown("---")

# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------

elif page == "About":
    st.title("About")
    st.markdown(
        """
        **Course:** CS 256 — Machine Learning (Prof. Mark Stamp, SJSU)

        **Dataset:** CIFAR-10 — 50k training / 10k test images, 32×32 RGB, 10 classes.

        **Models trained on Google Colab (A100 GPU):**

        | Model | Architecture highlights |
        |-------|------------------------|
        | VAE   | Encoder/Decoder, z_dim=256, beta-annealed KL (10-epoch warmup), AMP |
        | GAN   | DCGAN, G: 5× ConvTranspose2d, D: 4× Conv2d, BCEWithLogitsLoss, AMP |
        | Diffusion | UNet + sinusoidal time embedding, self-attention, EMA weights, cosine schedule T=1 000 |

        **Evaluation:** FID & IS computed via `torch-fidelity` against 10 000 generated samples.

        **Key finding:** Diffusion achieves the best FID (13.48) and IS (8.31) after 500 epochs,
        at the cost of significantly longer training. The GAN reaches competitive quality (FID 31.93)
        in just 100 epochs, while the VAE struggles with blurry outputs (FID 242).

        ---
        [GitHub](https://github.com/mrunalkotkar) |
        [Report](report.pdf)
        """
    )
