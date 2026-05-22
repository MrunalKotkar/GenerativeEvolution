"""
Streamlit portfolio — Generative Evolution
Run: streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Generative Evolution",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── paths ─────────────────────────────────────────────────────────────────────

SAMPLES  = Path("samples")
ASSETS   = Path("assets")
EVAL_DIR = ASSETS / "evaluation_graphs"
REPORT   = Path("report.pdf")
GITHUB   = "https://github.com/MrunalKotkar/GenerativeEvolution"

# ── constants ─────────────────────────────────────────────────────────────────

VAE_COL  = "#5DCAA5"
GAN_COL  = "#7F77DD"
DIFF_COL = "#378ADD"

BEST = {
    "vae":       {"fid": 242.15, "is_": 2.06,  "kid": 0.247, "epochs": 100, "time": "~2 h",   "batch": 256},
    "gan":       {"fid": 31.93,  "is_": 6.94,  "kid": 0.023, "epochs": 100, "time": "~1.5 h", "batch": 128},
    "diffusion": {"fid": 13.48,  "is_": 8.31,  "kid": 0.004, "epochs": 500, "time": "~18 h",  "batch": 64},
}

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── reduce default Streamlit side-margins, use full width ── */
.block-container {
  padding-top: 1.8rem !important;
  padding-left: 2.5rem !important;
  padding-right: 2.5rem !important;
  max-width: 1300px !important;
}

/* ── base readability ── */
body, p, li, span, div { font-size: 15px !important; }
h1 { font-size: 28px !important; font-weight: 600 !important; }
h2 { font-size: 22px !important; font-weight: 500 !important; }
h3 { font-size: 18px !important; font-weight: 500 !important; }

/* ── section label ── */
.sec-label {
  font-size: 12px !important; text-transform: uppercase;
  letter-spacing: .09em; color: #777; margin-bottom: .9rem; font-weight: 600;
}

/* ── hero ── */
.hero-title {
  font-size: 30px !important; font-weight: 600; line-height: 1.3;
  margin-bottom: .5rem; color: #111;
}
.hero-sub  { font-size: 16px !important; color: #555; line-height: 1.7; margin-bottom: .5rem; }
.hero-auth { font-size: 14px !important; color: #888; }

/* ── metric cards ── */
.mc {
  border: 1px solid #E5E7EB; border-radius: 14px;
  padding: 20px 22px; background: #FAFAFA;
}
.mc.best { border: 1.5px solid #BFDBFE; background: #EFF6FF; }
.mc-head {
  font-size: 14px !important; font-weight: 600; color: #444;
  display: flex; align-items: center; gap: 7px; margin-bottom: 10px;
}
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.mc-val  { font-size: 34px !important; font-weight: 600; margin-bottom: 3px; color: #111; }
.mc-sub  { font-size: 13px !important; color: #888; margin-bottom: 10px; }
.pill-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 6px; }
.pill {
  font-size: 12px !important; padding: 3px 10px; border-radius: 12px;
  border: 1px solid #D1D5DB; font-weight: 500;
}
.pill-good { background:#DCFCE7; color:#166534; border-color:#BBF7D0; }
.pill-mid  { background:#FEF9C3; color:#854D0E; border-color:#FDE047; }
.pill-bad  { background:#FEE2E2; color:#991B1B; border-color:#FECACA; }
.best-badge {
  margin-left: auto; font-size: 11px !important;
  background: #DBEAFE; color: #1E40AF; padding: 3px 8px;
  border-radius: 10px; font-weight: 600;
}

/* ── info cards inside tabs ── */
.icard {
  border: 1px solid #E5E7EB; border-radius: 12px;
  padding: 18px 20px; background: #fff; height: 100%;
}
.icard h4 { font-size: 15px !important; font-weight: 600; margin-bottom: 8px; }
.icard p  { font-size: 14px !important; color: #444; line-height: 1.7; }

/* ── sample image labels ── */
.img-label {
  font-size: 13px !important; font-weight: 600; color: #444;
  display: flex; align-items: center; gap: 6px; margin-bottom: 6px;
}

/* ── compare cards ── */
.ccard {
  border: 1px solid #E5E7EB; border-radius: 14px;
  padding: 20px; background: #FAFAFA; height: 100%;
}
.ccard.winner { border: 2px solid #BFDBFE; background: #EFF6FF; }
.ccard h4 {
  font-size: 15px !important; font-weight: 600; margin-bottom: 8px;
  display: flex; align-items: center; gap: 7px;
}
.ccard p  { font-size: 14px !important; color: #444; line-height: 1.65; margin-bottom: 10px; }
.tag-row  { display: flex; gap: 7px; flex-wrap: wrap; }
.tag {
  font-size: 12px !important; padding: 3px 10px; border-radius: 12px;
  border: 1px solid #E5E7EB; color: #555; background: #fff;
}

/* ── resource cards ── */
.rcard {
  border: 1px solid #E5E7EB; border-radius: 12px;
  padding: 20px 22px; background: #FAFAFA;
}
.rcard h4 { font-size: 15px !important; font-weight: 600; margin-bottom: 6px; }
.rcard p  { font-size: 14px !important; color: #555; line-height: 1.6; margin-bottom: 12px; }

/* ── divider ── */
hr.sec-div { border: none; border-top: 1px solid #E5E7EB; margin: 2rem 0 1.8rem; }

/* ── key findings list ── */
ul li { font-size: 15px !important; line-height: 1.75; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_metrics():
    dfs = {}
    for m in ("vae", "gan", "diffusion"):
        p = EVAL_DIR / f"{m}_metrics.csv"
        if p.exists():
            dfs[m] = pd.read_csv(p)
    return dfs


@st.cache_data
def load_img(path: Path):
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


@st.cache_data
def read_report():
    if REPORT.exists():
        return REPORT.read_bytes()
    return None


# ── chart helpers ─────────────────────────────────────────────────────────────

def line_chart_fid(dfs):
    fig = go.Figure()
    for key, label, color in [
        ("vae",       "VAE",              VAE_COL),
        ("gan",       "GAN (DCGAN)",      GAN_COL),
        ("diffusion", "Diffusion (DDPM)", DIFF_COL),
    ]:
        if key in dfs:
            df = dfs[key]
            fig.add_trace(go.Scatter(
                x=df["Epoch"], y=df["FID"],
                name=label, mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>Epoch: %{{x}}<br>FID: %{{y:.2f}}<extra></extra>",
            ))
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="FID (lower is better)",
        xaxis_title="Epoch",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#F0F0F0", zeroline=False, title_font=dict(size=14)),
        xaxis=dict(showgrid=False, title_font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=13)),
        font=dict(size=13),
    )
    return fig


def line_chart_is(dfs):
    fig = go.Figure()
    for key, label, color in [
        ("vae",       "VAE",              VAE_COL),
        ("gan",       "GAN (DCGAN)",      GAN_COL),
        ("diffusion", "Diffusion (DDPM)", DIFF_COL),
    ]:
        if key in dfs:
            df = dfs[key]
            fig.add_trace(go.Scatter(
                x=df["Epoch"], y=df["IS_mean"],
                name=label, mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>Epoch: %{{x}}<br>IS: %{{y:.2f}}<extra></extra>",
            ))
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="IS (higher is better)",
        xaxis_title="Epoch",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#F0F0F0", zeroline=False, title_font=dict(size=14)),
        xaxis=dict(showgrid=False, title_font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=13)),
        font=dict(size=13),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

dfs = load_metrics()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">Generative Evolution: Evolution of Image Generation<br>from Autoencoders to Diffusion Models</div>
<div class="hero-sub">Comparing VAE, DCGAN, and DDPM for unconditional image generation on CIFAR-10 (32×32).</div>
<div class="hero-auth">Mrunal Kotkar</div>
<hr class="sec-div">
""", unsafe_allow_html=True)

# ── Results at a glance ───────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Results at a glance</div>', unsafe_allow_html=True)

col_v, col_g, col_d = st.columns(3)

with col_v:
    st.markdown(f"""
    <div class="mc">
      <div class="mc-head"><span class="dot" style="background:{VAE_COL}"></span>VAE</div>
      <div class="mc-val">{BEST['vae']['fid']}</div>
      <div class="mc-sub">FID · lower is better</div>
      <div class="pill-row">
        <span class="pill pill-bad">IS {BEST['vae']['is_']}</span>
        <span class="pill pill-bad">KID {BEST['vae']['kid']}</span>
      </div>
    </div>""", unsafe_allow_html=True)

with col_g:
    st.markdown(f"""
    <div class="mc">
      <div class="mc-head"><span class="dot" style="background:{GAN_COL}"></span>GAN (DCGAN)</div>
      <div class="mc-val">{BEST['gan']['fid']}</div>
      <div class="mc-sub">FID · lower is better</div>
      <div class="pill-row">
        <span class="pill pill-mid">IS {BEST['gan']['is_']}</span>
        <span class="pill pill-mid">KID {BEST['gan']['kid']}</span>
      </div>
    </div>""", unsafe_allow_html=True)

with col_d:
    st.markdown(f"""
    <div class="mc best">
      <div class="mc-head"><span class="dot" style="background:{DIFF_COL}"></span>Diffusion (DDPM)<span class="best-badge">best</span></div>
      <div class="mc-val">{BEST['diffusion']['fid']}</div>
      <div class="mc-sub">FID · lower is better</div>
      <div class="pill-row">
        <span class="pill pill-good">IS {BEST['diffusion']['is_']}</span>
        <span class="pill pill-good">KID {BEST['diffusion']['kid']}</span>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

# ── Generated Samples — 3 columns, all visible at once ───────────────────────
st.markdown('<div class="sec-label">Generated Samples</div>', unsafe_allow_html=True)

col_v, col_g, col_d = st.columns(3)

with col_v:
    st.markdown(f'<div class="img-label"><span class="dot" style="background:{VAE_COL}"></span>VAE</div>', unsafe_allow_html=True)
    img = load_img(SAMPLES / "vae" / "vae.png")
    if img:
        st.image(img, use_container_width=True)
    else:
        st.info("Add `samples/vae/vae.png`")

with col_g:
    st.markdown(f'<div class="img-label"><span class="dot" style="background:{GAN_COL}"></span>GAN (DCGAN)</div>', unsafe_allow_html=True)
    img = load_img(SAMPLES / "gan" / "gan.png")
    if img:
        st.image(img, use_container_width=True)
    else:
        st.info("Add `samples/gan/gan.png`")

with col_d:
    st.markdown(f'<div class="img-label"><span class="dot" style="background:{DIFF_COL}"></span>Diffusion (DDPM)</div>', unsafe_allow_html=True)
    img = load_img(SAMPLES / "diffusion" / "diffusion.png")
    if img:
        st.image(img, use_container_width=True)
    else:
        st.info("Add `samples/diffusion/diffusion.png`")

st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

# ── Model Explorer ────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Model Explorer</div>', unsafe_allow_html=True)

tab_vae, tab_gan, tab_diff = st.tabs(["VAE", "GAN (DCGAN)", "Diffusion (DDPM)"])

with tab_vae:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="icard">
          <h4>Architecture</h4>
          <p>Convolutional encoder–decoder with <strong>256-dim latent space</strong>.
          4× Conv2d (k=4, s=2) encoder → FC heads (μ, log σ²).
          β-annealed KL warmup over 10 epochs. Loss: MSE + β·KL.</p>
          <div class="pill-row" style="margin-top:12px">
            <span class="pill">Batch {BEST['vae']['batch']}</span>
            <span class="pill">{BEST['vae']['epochs']} epochs</span>
            <span class="pill">{BEST['vae']['time']}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="icard">
          <h4>Strengths &amp; Weaknesses</h4>
          <p>Most stable training. Meaningful latent space enables smooth interpolation
          and latent traversal. Blurry outputs caused by pixel-level MSE averaging over the posterior.</p>
          <div class="pill-row" style="margin-top:12px">
            <span class="pill pill-good">Stable training</span>
            <span class="pill pill-good">Fast</span>
            <span class="pill pill-bad">Blurry output</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    arch = load_img(ASSETS / "VAE Architecture.png")
    if arch:
        with st.expander("View architecture diagram"):
            st.image(arch, use_container_width=True)

with tab_gan:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="icard">
          <h4>Architecture</h4>
          <p>DCGAN with <strong>128-dim noise input</strong>.
          Generator: 5× ConvTranspose2d (1→32px) + Tanh.
          Discriminator: 4× Conv2d → scalar logit.
          BCEWithLogitsLoss + AMP mixed precision.</p>
          <div class="pill-row" style="margin-top:12px">
            <span class="pill">Batch {BEST['gan']['batch']}</span>
            <span class="pill">{BEST['gan']['epochs']} epochs</span>
            <span class="pill">{BEST['gan']['time']}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="icard">
          <h4>Strengths &amp; Weaknesses</h4>
          <p>Sharp photo-realistic textures and fine detail. Fast inference (single forward pass).
          Training instability and risk of mode collapse require careful hyperparameter tuning.</p>
          <div class="pill-row" style="margin-top:12px">
            <span class="pill pill-good">Sharp images</span>
            <span class="pill pill-good">Fast inference</span>
            <span class="pill pill-mid">Training unstable</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    arch = load_img(ASSETS / "GAN Architecture.png")
    if arch:
        with st.expander("View architecture diagram"):
            st.image(arch, use_container_width=True)

with tab_diff:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="icard">
          <h4>Architecture</h4>
          <p>UNet with <strong>cosine schedule (T=1 000 steps)</strong>.
          Sinusoidal time embeddings, residual blocks, multi-head self-attention at 16×16 and 8×8.
          EMA (decay=0.9999) weights for sampling. AdamW + cosine LR schedule.</p>
          <div class="pill-row" style="margin-top:12px">
            <span class="pill">Batch {BEST['diffusion']['batch']}</span>
            <span class="pill">{BEST['diffusion']['epochs']} epochs</span>
            <span class="pill">{BEST['diffusion']['time']}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="icard">
          <h4>Strengths &amp; Weaknesses</h4>
          <p>State-of-the-art visual quality with excellent sample diversity. Stable, well-understood
          training. Slow sampling (1 000 denoising steps) and high GPU memory for the full UNet.</p>
          <div class="pill-row" style="margin-top:12px">
            <span class="pill pill-good">Best quality</span>
            <span class="pill pill-good">Stable training</span>
            <span class="pill pill-bad">Slow sampling</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    arch = load_img(ASSETS / "Diffusion Model Architecture.png")
    if arch:
        with st.expander("View architecture diagram"):
            st.image(arch, use_container_width=True)

st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

# ── Training metric evolution ─────────────────────────────────────────────────
st.markdown('<div class="sec-label">Training Metric Evolution</div>', unsafe_allow_html=True)

metric_choice = st.radio(
    "Metric", ["FID (lower is better)", "IS (higher is better)"],
    horizontal=True, label_visibility="collapsed",
)
if metric_choice.startswith("FID"):
    st.plotly_chart(line_chart_fid(dfs), use_container_width=True)
else:
    st.plotly_chart(line_chart_is(dfs), use_container_width=True)

st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

# ── Model Comparison ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Model Comparison</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown(f"""
    <div class="ccard">
      <h4><span class="dot" style="background:{VAE_COL};width:11px;height:11px;border-radius:50%;display:inline-block"></span>VAE</h4>
      <p>Best for <strong>representation learning</strong> and latent space analysis.
      Use when training stability matters more than image fidelity.</p>
      <div class="tag-row">
        <span class="tag">Stable</span>
        <span class="tag">Fast (2 h)</span>
        <span class="tag">Blurry</span>
      </div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="ccard">
      <h4><span class="dot" style="background:{GAN_COL};width:11px;height:11px;border-radius:50%;display:inline-block"></span>GAN</h4>
      <p>Best <strong>quality-to-speed trade-off</strong>. Reaches competitive FID in 100 epochs.
      Needs careful tuning to avoid mode collapse.</p>
      <div class="tag-row">
        <span class="tag">Sharp</span>
        <span class="tag">Fast inference</span>
        <span class="tag">Needs tuning</span>
      </div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="ccard winner">
      <h4><span class="dot" style="background:{DIFF_COL};width:11px;height:11px;border-radius:50%;display:inline-block"></span>Diffusion <span class="best-badge">winner</span></h4>
      <p>State-of-the-art quality with diverse samples. High compute cost —
      best when GPU resources are available and quality is paramount.</p>
      <div class="tag-row">
        <span class="tag">Best quality</span>
        <span class="tag">Diverse</span>
        <span class="tag">Slow (18 h)</span>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

# ── Key findings ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Key Findings</div>', unsafe_allow_html=True)
st.markdown("""
- **Diffusion** achieves the best FID (13.48) and IS (8.31) after 500 epochs, but costs ~12× more training time than the GAN.
- **GAN** delivers competitive quality (FID 31.93) in just 100 epochs — the best quality-to-training-time trade-off.
- **VAE** generates blurry images (FID 242) because pixel-level MSE averages over the posterior; IS barely exceeds 2 at epoch 100.
- EMA smoothing in the diffusion model yields noticeably sharper samples than raw model weights.
- Diffusion FID drops sharply between epochs 250–400 (349 → 16), then plateaus — most of the compute goes toward the last ~4 FID points.
""")

st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

# ── Resources ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Resources</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="medium")

with c1:
    st.markdown("""
    <div class="rcard">
      <h4>📄 Project Report</h4>
      <p>Full write-up with methodology, model details, evaluation setup,
      ablations, and discussion of failure modes.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    report_bytes = read_report()
    if report_bytes:
        st.download_button(
            label="Download Report (PDF)",
            data=report_bytes,
            file_name="Generative_Evolution_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("report.pdf not found.")

with c2:
    st.markdown(f"""
    <div class="rcard">
      <h4>💻 Source Code</h4>
      <p>Self-contained training scripts for each model — no Colab required.
      Each module exposes <code>train()</code> and <code>generate()</code> entry points.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.link_button(
        "View on GitHub",
        url=GITHUB,
        use_container_width=True,
    )
