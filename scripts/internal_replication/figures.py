"""Main figures for the intervention-coverage manuscript, from published numeric tables."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("artifacts")
out = Path("paper")
out.mkdir(exist_ok=True)
primary = json.loads((root / "qwen-predictive-coverage.json").read_text())["rows"]
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)
colors = ["#153e75", "#a13c21", "#238378"]
methods = [
    ("ILS", "ILS"),
    ("horizon_scaled_identity", "Direct residual"),
    ("zero_prior_ridge", "Ridge (zero prior)"),
]


def panel(ax, rows, g, keys):
    rows = sorted((r for r in rows if r["gamma"] == g), key=lambda r: r["layer"])
    x = [r["layer"] for r in rows]
    ax.axhline(0, color=".65", lw=0.8)
    for (key, label), color in zip(keys, colors):
        values = [r["metrics"][key] for r in rows]
        ax.plot(
            x,
            [v["estimate"] for v in values],
            color=color,
            label=label,
            lw=1.6,
            marker="o",
            ms=2.2,
        )
        ci = np.array([v["ci"] for v in values])
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.1, linewidth=0)
    ax.axvline(x[-1], ls=":", lw=0.8, color=".4")
    ax.set_title("γ = " + str(g))
    ax.set_xlabel("Block output")


fig, axes = plt.subplots(1, 4, figsize=(10.3, 2.9), sharey=True)
for ax, g in zip(axes, [0, 0.5, 0.9, 1]):
    panel(ax, primary, g, methods)
axes[0].set_ylabel("Held-out R²")
axes[0].legend(fontsize=7, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(out / "primary-depth.png")
plt.close(fig)
rep = json.loads((root / "qwen35-predictive-coverage.json").read_text())
fig, axes = plt.subplots(2, 4, figsize=(10.3, 5.2), sharey=True)
for group, axs in zip(rep["results"], axes):
    for ax, g in zip(axs, [0, 0.5, 0.9, 1]):
        panel(ax, group["rows"], g, [("ILS", "Identity-prior ILS"), *methods[1:]])
        ax.set_xticks([7, 15, 23, 30])
        ax.set_yscale("symlog", linthresh=1)
    axs[0].set_ylabel(group["model"].upper() + " held-out R²")
axes[0, 0].legend(fontsize=7, loc="best", frameon=False)
fig.tight_layout()
fig.savefig(out / "replication-depth.png")
plt.close(fig)

strict = json.loads((root / "qwen-future-only.json").read_text())["rows"]
fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), sharey=True)
for ax, model in zip(axes, ["0.5b", "4b", "9b"]):
    for gamma, col in zip([0.5, 0.9, 1], colors):
        rs = sorted(
            (r for r in strict if r["model"] == model and r["gamma"] == gamma),
            key=lambda r: r["layer"],
        )
        x = [r["layer"] for r in rs]
        ax.plot(
            x,
            [r["ridge"]["estimate"] for r in rs],
            color=col,
            marker="o",
            ms=2.2,
            lw=1.5,
            label=f"γ = {gamma}",
        )
        ci = np.array([r["ridge"]["ci"] for r in rs])
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=col, alpha=0.1, linewidth=0)
    ax.axhline(0, color=".65", lw=0.8)
    ax.set(xlabel="Block output", title=model.upper())
axes[0].set_ylabel("Later-only held-out R²")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(out / "future-depth.png")
plt.close(fig)
