from os import XATTR_SIZE_MAX
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import matplotlib.ticker as ticker


def plt_box(
    df: pd.DataFrame,
    x,
    y,
    whis=[10, 90],
    fontsize=24,
    hue=None,
    dodge=True,
    title="",
    tick_step=None,
    **kwargs,
) -> str:
    tmp = df.sort_values(by=x)
    plt.figure(figsize=(1.5 * len(df[x].unique()), 6))
    ax = sns.boxplot(
        data=tmp,
        x=x,
        y=y,
        hue=hue,
        patch_artist=True,
        showfliers=False,
        whis=whis,
        width=0.4,
        # gap=0.3,
        showmeans=True,
        dodge=dodge,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
        **kwargs,
    )
    for _, patch in enumerate(ax.patches):
        patch.set_linewidth(1.25)
        patch.set_edgecolor("black")

    for line in ax.lines:
        if line.get_linestyle() == "-":
            line.set_color("black")
            line.set_linewidth(1.25)

    if hue is None:
        for _, patch in enumerate(ax.patches):
            patch.set_facecolor("lightblue")

        ax.patches[-1].set_facecolor("lightgreen")
        ax.patches[-2].set_facecolor("lightgreen")

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    # if tmp[y].max() <= 1:
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    if tick_step is not None:
        ymin, ymax = ax.get_ylim()
        ticks_up = np.arange(1, ymax, tick_step)
        ticks_down = np.arange(1, ymin, -tick_step)
        new_ticks = np.unique(np.concatenate([ticks_up, ticks_down]))
        ax.set_yticks(new_ticks)

    plt.title(title, fontsize=fontsize)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(x, fontsize=fontsize)
    plt.ylabel(y, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    if labels and max(len(lbl) for lbl in labels) > 8:
        plt.xticks(rotation=45, ha="right", fontsize=fontsize)
    else:
        plt.xticks(fontsize=fontsize)

    buf = io.BytesIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    svg_base64 = base64.b64encode(buf.read()).decode("utf-8")
    md = f"![plot](data:image/svg+xml;base64,{svg_base64})"
    return md
