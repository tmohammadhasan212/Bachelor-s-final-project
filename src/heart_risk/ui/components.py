"""
Reusable visual UI components, gauges, and clinical charts for Streamlit dashboard.
"""

from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def render_risk_gauge(risk_pct: float, category: str):
    """Render a clean clinical risk gauge with color coding."""
    color_map = {
        "Critical Risk": "#dc2626",
        "High Risk": "#ea580c",
        "Moderate Risk": "#d97706",
        "Low Risk": "#16a34a",
    }
    color = color_map.get(category, "#2563eb")

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    border-radius: 12px; padding: 20px; color: #ffffff; text-align: center;
                    border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">
                Estimated Cardiovascular Risk
            </div>
            <div style="font-size: 48px; font-weight: 800; color: {color}; margin: 8px 0;">
                {risk_pct:.1f}%
            </div>
            <div style="display: inline-block; background-color: {color}; color: #ffffff;
                        padding: 4px 14px; border-radius: 9999px; font-size: 14px; font-weight: 700;">
                {category.upper()}
            </div>
            <div style="margin-top: 14px; height: 8px; background: #334155; border-radius: 4px; overflow: hidden;">
                <div style="width: {min(max(risk_pct, 0.0), 100.0)}%; height: 100%; background: {color}; border-radius: 4px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_waterfall_chart(
    contributions: List[Any],
    base_risk: float,
    predicted_risk: float,
) -> plt.Figure:
    """Create a clean horizontal waterfall/bar chart for feature attribution."""
    features = [c.display_name for c in reversed(contributions)]
    values = [c.contribution * 100.0 for c in reversed(contributions)]
    colors = [
        "#dc2626" if v > 0 else "#16a34a" for v in values
    ]  # Red increases risk, Green decreases

    fig, ax = plt.subplots(figsize=(8, max(len(features) * 0.45, 3.5)))
    y_pos = np.arange(len(features))

    bars = ax.barh(y_pos, values, color=colors, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Impact on Predicted Risk Percentage Points (±%)", fontsize=10)
    ax.set_title(
        f"Feature Risk Attribution (Base: {base_risk*100:.1f}% → Final: {predicted_risk*100:.1f}%)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.3 if width >= 0 else -0.3
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.1f}%",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold",
            color="#1e293b",
        )

    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_dca_curve(dca_dict: Dict[str, Any]) -> plt.Figure:
    """Plot Decision Curve Analysis (Net Benefit curve)."""
    thresholds = dca_dict["thresholds"]
    nb_model = dca_dict["model_net_benefit"]
    nb_all = dca_dict["treat_all_net_benefit"]
    nb_none = dca_dict["treat_none_net_benefit"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        thresholds,
        nb_model,
        label="Heart Risk CDS Model",
        color="#0284c7",
        linewidth=2.5,
    )
    ax.plot(
        thresholds,
        nb_all,
        label="Treat All (Default Intervention)",
        color="#64748b",
        linestyle="--",
        linewidth=1.8,
    )
    ax.plot(
        thresholds,
        nb_none,
        label="Treat None (Net Benefit = 0)",
        color="#0f172a",
        linestyle=":",
        linewidth=1.5,
    )

    ax.set_xlim([0.05, 0.85])
    ax.set_ylim([-0.05, max(max(nb_model), 0.5) + 0.05])
    ax.set_xlabel("Clinical Decision Threshold Probability ($p_t$)", fontsize=10)
    ax.set_ylabel("Net Clinical Benefit", fontsize=10)
    ax.set_title(
        "Decision Curve Analysis (Clinical Utility & Net Benefit)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig
