# -*- coding: utf-8 -*-
"""
PhD Coating Window Analyzer — Streamlit (Lightweight)
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MultipleLocator, AutoMinorLocator
from datetime import datetime

st.set_page_config(page_title="Coating Window Analyzer", page_icon="🧪", layout="wide")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.linewidth": 2.2,
    "xtick.major.width": 1.8, "ytick.major.width": 1.8,
    "xtick.minor.width": 1.2, "ytick.minor.width": 1.2,
    "xtick.direction": "in",  "ytick.direction": "in",
    "xtick.major.size": 10,   "ytick.major.size": 10,
    "xtick.minor.size": 5,    "ytick.minor.size": 5,
    "xtick.top": True,        "ytick.right": True,
    "legend.frameon": True,   "lines.linewidth": 3.5,
    "svg.fonttype": "none"
})

COLORS = ['#CC6677','#332288','#88CCEE','#44AA99','#117733','#999933','#DDCC77','#882255','#AA4499']

# ── Session state ──────────────────────────────────────────────────────────────
if "points" not in st.session_state:
    st.session_state.points = []
if "counter" not in st.session_state:
    st.session_state.counter = 1

# ── Figure ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_model_curve():
    G = np.linspace(1.01, 500, 1000)
    return G, 0.65 * (2 / (G - 1)) ** 1.5

def build_figure(points):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel("Dimensionless Gap [$H_0 / t_{wet}$]", fontsize=16, fontweight='bold', labelpad=12)
    ax.set_ylabel("Capillary Number [$Ca$]", fontsize=16, fontweight='bold', labelpad=12)
    ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
    ax.grid(True, which='both', linestyle=':', alpha=0.3, color='gray')
    ax.set_title("Visco-Capillary Stability Window", fontsize=15, pad=15, fontweight='bold')

    G, Ca = get_model_curve()
    ax.plot(G, Ca, color='#332288', label="VC Model Limit", linewidth=4, alpha=0.8)

    max_x = max((p["H0/twet"] for p in points), default=480)
    ax.set_xlim(-50, max(500, max_x * 1.2))
    ax.set_ylim(1e-4, 10)

    for p in points:
        color = COLORS[(p["ID"] - 1) % len(COLORS)]
        ax.plot(p["H0/twet"], p["Ca"], 'o', color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=1.2, label=f"Point {p['ID']}")

    ax.legend(loc="upper right", fontsize=9, framealpha=1.0, edgecolor='black', ncol=2)
    fig.tight_layout()
    return fig

# ── Export helpers ─────────────────────────────────────────────────────────────
def fig_to_bytes(fig, fmt, dpi=400):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.read()

def table_to_txt(points):
    lines = ["COATING STABILITY REPORT",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "-" * 110,
             f"{'Pt':<4} | {'Visc':<8} | {'Speed':<8} | {'Sigma':<8} | {'Q':<8} | {'Width':<8} | {'Gap':<8} | {'Ca':<10} | {'t_wet':<8} | {'G_dim':<8}",
             "-" * 110]
    for p in points:
        lines.append(f"{p['ID']:<4} | {p['Visc [mPa.s]']:<8.2f} | {p['Speed [m/min]']:<8.2f} | "
                     f"{p['Sigma [mN/m]']:<8.2f} | {p['Q [mL/min]']:<8.2f} | {p['Width [cm]']:<8.2f} | "
                     f"{p['Gap [um]']:<8.2f} | {p['Ca']:<10.4e} | {p['t_wet [um]']:<8.2f} | {p['H0/twet']:<8.2f}")
    lines.append("-" * 110)
    return "\n".join(lines)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🧪 Coating Window Analyzer")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Parameters")

    # Wrap ALL inputs in a form — nothing runs until Submit is clicked
    with st.form("input_form", clear_on_submit=False):
        mu_val    = st.number_input("Viscosity [mPa·s]",      min_value=0.01, value=50.0,  step=1.0)
        v_mpm     = st.number_input("Speed [m/min]",           min_value=0.01, value=10.0,  step=0.5)
        sigma_val = st.number_input("Surface Tension [mN/m]",  min_value=0.01, value=30.0,  step=0.5)
        q_val     = st.number_input("Flow Rate [mL/min]",      min_value=0.01, value=5.0,   step=0.1)
        w_val     = st.number_input("Width [cm]",              min_value=0.01, value=10.0,  step=0.5)
        g_val     = st.number_input("Gap [µm]",                min_value=0.01, value=100.0, step=5.0)
        submitted = st.form_submit_button("➕ ADD POINT", use_container_width=True, type="primary")

    if submitted:
        v_mps   = v_mpm / 60.0
        ca_val  = (mu_val * 1e-3 * v_mps) / (sigma_val * 1e-3)
        twet_um = (100.0 * q_val) / (w_val * v_mpm)
        x_val   = g_val / twet_um

        st.session_state.points.append({
            "ID": st.session_state.counter,
            "Visc [mPa.s]": mu_val, "Speed [m/min]": v_mpm,
            "Sigma [mN/m]": sigma_val, "Q [mL/min]": q_val,
            "Width [cm]": w_val, "Gap [um]": g_val,
            "Ca": ca_val, "t_wet [um]": twet_um, "H0/twet": x_val
        })
        st.session_state.counter += 1
        st.success(f"Point {st.session_state.counter - 1} added  —  Ca = {ca_val:.4e},  t_wet = {twet_um:.2f} µm")

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.points = []
        st.session_state.counter = 1
        st.rerun()

with col_right:
    fig = build_figure(st.session_state.points)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if st.session_state.points:
        st.subheader("Export")
        c1, c2, c3 = st.columns(3)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        c1.download_button("⬇️ PNG", data=fig_to_bytes(fig, "png"), file_name=f"Coating_{ts}.png", mime="image/png", use_container_width=True)
        c2.download_button("⬇️ SVG", data=fig_to_bytes(fig, "svg"), file_name=f"Coating_{ts}.svg", mime="image/svg+xml", use_container_width=True)
        c3.download_button("⬇️ TXT", data=table_to_txt(st.session_state.points), file_name=f"Coating_{ts}.txt", mime="text/plain", use_container_width=True)

        st.subheader("Data Table")
        st.dataframe(st.session_state.points, use_container_width=True, hide_index=True)
