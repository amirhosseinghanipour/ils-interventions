"""Draw the matched depth profiles directly from committed numerical results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from score import read


def figure(path, output):
    fig, axes = plt.subplots(2,2, figsize=(7.2,4.45), sharex=True, sharey="row", layout="constrained")
    for j,size in enumerate(("4b","9b")):
        cal = read(path/f"dense-centering-{size}.json")
        scan = read(path/f"{size}.json")["blocks"]["depth"]
        x = np.arange(15,24)
        for name, label, color, mark in (
            ("carrier","Neutral-word mean","#245884","o"),
            ("crossfit","Other task prompts","#a24420","s")):
            y = [100 * cal[str(k)][name]["correct"]/128 for k in x]
            axes[0,j].plot(x,y,marker=mark,ms=3.2,lw=1.5,color=color,label=label)
        axes[0,j].set_title(f"Qwen3.5-{size.upper()}", fontsize=11)
        xs = np.arange(16,24)
        vals = np.array([scan[str(k)]["mean"] for k in xs])
        intervals = np.array([scan[str(k)]["ci95"] for k in xs])
        axes[1,j].errorbar(xs,vals,yerr=np.abs(intervals.T-vals),color="#245884",marker="o",ms=3.2,lw=1.3,capsize=2)
        axes[1,j].scatter(19,vals[3],marker="o",s=27,color="#a24420",zorder=5)
        axes[1,j].axhline(0,lw=.7,color="#888888")
        axes[1,j].set(xlabel="Block (zero-based)", xticks=x)
        for ax in axes[:,j]:
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",lw=.5,color="#e5e5e5")
            ax.set_axisbelow(True)
    axes[0,0].set(ylabel="Food accuracy (%)",ylim=(-5,106),yticks=[0,25,50,75,100])
    axes[0,0].legend(loc="upper left",fontsize=7,frameon=False)
    axes[1,0].set_ylabel("Margin gain (logits)")
    fig.savefig(output,bbox_inches="tight",metadata={"Creator":"ILS-lens"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scores",type=Path)
    p.add_argument("output",type=Path)
    a = p.parse_args()
    figure(a.scores,a.output)
