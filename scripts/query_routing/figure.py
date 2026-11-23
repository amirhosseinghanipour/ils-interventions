"""Draw the two comparisons from the compact result files."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
results = ROOT / "artifacts/query-routing"


def read(where):
    return json.loads((results / where).read_text())


def dot(ax,x,record,color,marker):
    y=record.get("estimate",record.get("mean"))
    lo,hi=record["interval"]
    ax.vlines(x,lo,hi,linewidth=1.35,color=color)
    ax.plot([x,x],[lo,hi],linestyle="none",marker="_",markersize=5,color=color)
    ax.plot(x,y,marker=marker,markersize=5,color=color)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    output=parser.parse_args().output
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,
                         "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(6.65,2.65),layout="constrained")
    ledger="#24647d"; parcel="#ad592d"
    for ax,metric,heading in zip(axes,("full_edit_behavior","r2"),
                                 ("Answer-margin contrast","Frozen residual prediction")):
        ax.axhline(0,color="#777777",linewidth=.7)
        ax.set_title(heading,loc="left",weight="bold",fontsize=10)
        ax.grid(axis="y",color="#dddddd",linewidth=.55);ax.set_axisbelow(True)
        for i,size in enumerate(("4b","9b")):
            for j,(prefix,color,marker) in enumerate((("",ledger,"o"),("parcels/",parcel,"s"))):
                stats=read(prefix+size+"/full.json")[metric]
                keys=("difference","random_difference") if metric=="full_edit_behavior" else ("shared","conditional")
                for k,key in enumerate(keys):
                    x=2.65*i+k+(.115 if j else -.115)
                    dot(ax,x,stats[key],color,marker)
        labels=(["Donor","Random"] if metric=="full_edit_behavior" else ["Shared","Cond."])*2
        ax.set_xticks([0,1,2.65,3.65],labels)
        for x,title in ((.5,"4B"),(3.15,"9B")):
            ax.annotate(title,xy=(x,-.18),xycoords=("data","axes fraction"),ha="center",va="top",weight="bold")
        ax.set_xlim(-.45,4.1)
    axes[0].set_ylabel("Relevant minus irrelevant (logits)")
    axes[1].set_ylabel("Held-out residual $R^2$")
    for label,color,marker in (("New color prompts",ledger,"o"),("Unseen objects",parcel,"s")):
        axes[1].plot([],[],marker,color=color,label=label,markersize=4.5)
    axes[1].legend(frameon=False,fontsize=7.3,loc="best",handlelength=.85)
    fig.savefig(output,bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__=="__main__": main()
