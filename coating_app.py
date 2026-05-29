# -*- coding: utf-8 -*-
"""
PhD Coating Window Analyzer — Streamlit Web App
Ported from CoatingWindow.py v1.6
"""

import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MultipleLocator, AutoMinorLocator
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Coating Window Analyzer",
    page_icon="🧪",
    layout="wide"
)

# =============================================================================
# MATPLOTLIB STYLE
# =============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 2.2,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.minor.width": 1.2,
    "ytick.minor.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "lines.linewidth": 3.5,
    "svg.fonttype": "none"
})

SCIENTIFIC_COLORS = [
    '#CC6677', '#332288', '#88CCEE', '#44AA99',
    '#117733', '#999933', '#DDCC77', '#882255', '#AA4499'
]

# =============================================================================
# SESSION STATE INIT
# =============================================================================
if "points" not in st.session_state:
    st.session_state.points = []   # list of dicts
if "counter" not in st.session_state:
    st.session_state.counter = 1

# =============================================================================
# FIGURE BUILDER
# =============================================================================
def build_figure(points):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Axes style
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel("Dimensionless Gap [$H_0 / t_{wet}$]", fontsize=16, fontweight='bold', labelpad=12)
    ax.set_ylabel("Capillary Number [$Ca$]", fontsize=16, fontweight='bold', labelpad=12)
    ax.tick_params(axis='both', which='major', labelsize=12, pad=8)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid(True, which='both', linestyle=':', alpha=0.3, color='gray')
    ax.set_title("Visco-Capillary Stability Window", fontsize=15, pad=15, fontweight='bold')

    # Model limit curve
    G_range = np.linspace(-49, 500, 1000)
    Ca_model = 0.65 * (2 / (G_range - 1)) ** 1.5
    Ca_model = np.where(G_range > 1, Ca_model, np.nan)
    ax.plot(G_range, Ca_model, color='#332288', label="VC Model Limit", linewidth=4, alpha=0.8)

    # Dynamic x limit
    max_x = max((p["H0/twet"] for p in points), default=480)
    ax.set_xlim(-50, max(500, max_x * 1.2))
    ax.set_ylim(1e-4, 10)

    # Plot recorded points
    for p in points:
        color = SCIENTIFIC_COLORS[(p["ID"] - 1) % len(SCIENTIFIC_COLORS)]
        ax.plot(p["H0/twet"], p["Ca"], 'o',
                color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=1.2,
                label=f"Point {p['ID']}")

    ax.legend(loc="upper right", fontsize=9, framealpha=1.0, edgecolor='black', ncol=2)
    fig.tight_layout()
    return fig

# =============================================================================
# EXPORT HELPERS
# =============================================================================
def fig_to_bytes(fig, fmt, dpi=400):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.read()

def table_to_txt(points):
    lines = []
    lines.append("COATING STABILITY REPORT")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-" * 115)
    header = f"{'Pt':<4} | {'Visc':<8} | {'Speed':<8} | {'Sigma':<8} | {'Q':<8} | {'Width':<8} | {'Gap':<8} | {'Ca':<10} | {'t_wet':<8} | {'G_dim':<8}"
    lines.append(header)
    lines.append("-" * 115)
    for p in points:
        row = (f"{p['ID']:<4} | {p['Visc [mPa.s]']:<8.2f} | {p['Speed [m/min]']:<8.2f} | "
               f"{p['Sigma [mN/m]']:<8.2f} | {p['Q [mL/min]']:<8.2f} | {p['Width [cm]']:<8.2f} | "
               f"{p['Gap [um]']:<8.2f} | {p['Ca']:<10.4e} | {p['t_wet [um]']:<8.2f} | {p['H0/twet']:<8.2f}")
        lines.append(row)
    lines.append("-" * 115)
    return "\n".join(lines)

# =============================================================================
# UI LAYOUT
# =============================================================================
st.title("🧪 PhD Coating Window Analyzer")
st.caption("Visco-Capillary Stability Window — Web Edition")

col_inputs, col_plot = st.columns([1, 2.2], gap="large")

# ── LEFT: Inputs ──────────────────────────────────────────────────────────────
with col_inputs:
    st.subheader("Input Parameters")

    mu_val    = st.number_input("Viscosity [mPa·s]",        min_value=0.01, value=50.0,  step=1.0)
    v_mpm     = st.number_input("Speed [m/min]",             min_value=0.01, value=10.0,  step=0.5)
    sigma_val = st.number_input("Surface Tension [mN/m]",    min_value=0.01, value=30.0,  step=0.5)
    q_val     = st.number_input("Flow Rate [mL/min]",        min_value=0.01, value=5.0,   step=0.1)
    w_val     = st.number_input("Width [cm]",                min_value=0.01, value=10.0,  step=0.5)
    g_val     = st.number_input("Gap [µm]",                  min_value=0.01, value=100.0, step=5.0)

    st.divider()

    # Calculated preview
    v_mps    = v_mpm / 60.0
    ca_val   = (mu_val * 1e-3 * v_mps) / (sigma_val * 1e-3)
    twet_um  = (100.0 * q_val) / (w_val * v_mpm) if (w_val * v_mpm) != 0 else 0
    x_val    = g_val / twet_um if twet_um > 0 else 0

    st.markdown("**Live Preview**")
    m1, m2 = st.columns(2)
    m1.metric("Ca", f"{ca_val:.4e}")
    m2.metric("t_wet [µm]", f"{twet_um:.2f}")

    st.divider()

    btn_add   = st.button("➕ ADD POINT",   use_container_width=True, type="primary")
    btn_reset = st.button("🔄 RESET GRAPH", use_container_width=True)

    if btn_add:
        if ca_val <= 0 or twet_um <= 0:
            st.error("Ca or t_wet invalid. Check viscosity or flow rate.")
        else:
            st.session_state.points.append({
                "ID": st.session_state.counter,
                "Visc [mPa.s]": mu_val,
                "Speed [m/min]": v_mpm,
                "Sigma [mN/m]": sigma_val,
                "Q [mL/min]": q_val,
                "Width [cm]": w_val,
                "Gap [um]": g_val,
                "Ca": ca_val,
                "t_wet [um]": twet_um,
                "H0/twet": x_val
            })
            st.session_state.counter += 1
            st.success(f"Point {st.session_state.counter - 1} added!")

    if btn_reset:
        st.session_state.points = []
        st.session_state.counter = 1
        st.info("Graph reset.")

# ── RIGHT: Plot + Export ───────────────────────────────────────────────────────
with col_plot:
    fig = build_figure(st.session_state.points)
    st.pyplot(fig, use_container_width=True)

    # Export
    if st.session_state.points:
        st.subheader("📥 Export")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ec1, ec2, ec3 = st.columns(3)

        with ec1:
            st.download_button(
                "⬇️ PNG (400 DPI)",
                data=fig_to_bytes(fig, "png", dpi=400),
                file_name=f"CoatingAnalysis_{timestamp}.png",
                mime="image/png",
                use_container_width=True
            )
        with ec2:
            st.download_button(
                "⬇️ SVG (Vector)",
                data=fig_to_bytes(fig, "svg"),
                file_name=f"CoatingAnalysis_{timestamp}.svg",
                mime="image/svg+xml",
                use_container_width=True
            )
        with ec3:
            st.download_button(
                "⬇️ TXT (Data Table)",
                data=table_to_txt(st.session_state.points),
                file_name=f"CoatingAnalysis_{timestamp}_Data.txt",
                mime="text/plain",
                use_container_width=True
            )

        # Data table
        st.subheader("📊 Recorded Points")
        st.dataframe(
            st.session_state.points,
            use_container_width=True,
            hide_index=True
        )
