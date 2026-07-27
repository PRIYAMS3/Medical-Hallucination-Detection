import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def card(ax, x, y, w, h, title, value):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#263244",
        facecolor="#121a27",
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.08, title, color="#8fa3bf", fontsize=10, family="DejaVu Sans")
    ax.text(x + 0.02, y + 0.08, value, color="#f5f8ff", fontsize=18, family="DejaVu Sans", weight="bold")


def main():
    fig = plt.figure(figsize=(12, 6.75), dpi=140, facecolor="#0b0f17")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.93, "AI Smart Inventory Management", color="#f5f8ff", fontsize=18, weight="bold")
    ax.text(0.04, 0.89, "Streamlit dashboard preview", color="#8fa3bf", fontsize=10)

    card(ax, 0.04, 0.68, 0.2, 0.16, "Predicted Demand", "12,480")
    card(ax, 0.27, 0.68, 0.2, 0.16, "Stock Risk", "Low")
    card(ax, 0.50, 0.68, 0.2, 0.16, "Promo Impact", "+8.2%")

    chart_patch = FancyBboxPatch(
        (0.04, 0.10),
        0.66,
        0.52,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#263244",
        facecolor="#121a27",
    )
    ax.add_patch(chart_patch)

    chart = fig.add_axes([0.08, 0.17, 0.58, 0.40], facecolor="#121a27")
    x = np.arange(1, 13)
    actual = np.array([105, 98, 110, 120, 117, 125, 132, 128, 136, 142, 150, 158])
    pred = np.array([102, 100, 108, 118, 119, 123, 130, 129, 134, 140, 149, 156])
    chart.plot(x, actual, color="#22d3ee", linewidth=2.5, label="Actual")
    chart.plot(x, pred, color="#67e8a7", linewidth=2.5, linestyle="--", label="Predicted")
    chart.set_title("Sales vs Forecast", color="#c8d5e6", fontsize=11, pad=10)
    chart.grid(alpha=0.20, color="#5d6f85")
    chart.tick_params(colors="#8fa3bf")
    for spine in chart.spines.values():
        spine.set_color("#263244")
    chart.legend(facecolor="#121a27", edgecolor="#263244", labelcolor="#c8d5e6", fontsize=9)

    side = FancyBboxPatch(
        (0.74, 0.10),
        0.22,
        0.52,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#263244",
        facecolor="#121a27",
    )
    ax.add_patch(side)
    ax.text(0.77, 0.56, "Prediction", color="#8fa3bf", fontsize=10)
    ax.text(0.77, 0.49, "Store: 12", color="#f5f8ff", fontsize=11)
    ax.text(0.77, 0.43, "Family: Dairy", color="#f5f8ff", fontsize=11)
    ax.text(0.77, 0.37, "Promotion: Yes", color="#f5f8ff", fontsize=11)
    ax.text(0.77, 0.29, "Forecast", color="#8fa3bf", fontsize=10)
    ax.text(0.77, 0.20, "1,248 units", color="#67e8a7", fontsize=20, weight="bold")

    fig.savefig("dashboard_mock.png", dpi=140, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
