import re
import io
from typing_extensions import Writer
import plotly.express as px
import plotly.graph_objects as go
import os
from glob import glob
from pathlib import Path
import pprint
from typing import Final
import numpy as np
import seaborn as sns
import age_parser as age
import other_parser as other

import pandas as pd
from common import extract_desc, sort_key
from docs_writer import DocsWriter
from outputs_parser import GetResult
from plotly_wrapper import Box, Scatter, Box_2
import matplotlib.pyplot as plt

from matplotlib_wrapper import heat_map, plt_box

OUTPUT_PATH: Final[str] = "../docs/"
DATA_PATH: Final[str] = "../simulation_results/hashed/"

MEASUREMENTS = [
    "Relative Miss Ratio [LRU]",
    "Relative Promotion [LRU]",
    "Relative Miss Ratio [FR]",
    "Relative Promotion [FR]",
    "Abs. D. Miss Ratio",
]

ALGO: dict[str, str | tuple[str, int]] = {
    "FIFO": "fifo",
    "LRU": "lru",
    "FR": "clock",
    "D-FR": "dclock",
}

BASE_ALGO = ["FR"]

PALETTE = ["lightblue", "lightgreen", "lightpink", "purple", "gray"]
PALETTE = "pastel"


def WriteAggregatePLT(
    writer: DocsWriter,
    df: pd.DataFrame,
    desc: str = "",
):
    df = df.reset_index()
    for key in MEASUREMENTS:
        title = f"{desc}"
        writer.Write(f"### {key}")
        fig = plt_box(
            df,
            y=key,
            x="P",
            hue="Algorithm",
            title=title,
            palette=PALETTE,
            tick_step=0.01 if "Miss" in key else None,
        )
        writer.Write(fig)


def WriteAggregate(
    writer: DocsWriter,
    df: pd.DataFrame,
    desc: str = "",
):
    writer.Write(f"## Aggregate Results for {desc}")
    for key in MEASUREMENTS:
        title = f"{key} for {desc}"
        writer.Write(f"### {key}")
        fig = Box(
            df,
            y=key,
            x="Algorithm",
            title=title,
        )
        writer.WriteFig(fig)


def AdditionalProcessing(df: pd.DataFrame, compared_algo="FR"):
    df["Trace Group"] = df["Trace Path"].apply(lambda x: Path(x).parts[0])
    df["Miss"] = df["Request"] - df["Hit"]
    # for key in ["Hit", "Miss Ratio", "Reinserted"]:
    #     base = (
    #         df.set_index("Algorithm")
    #         .loc[compared_algo]
    #         .set_index(["Cache Size", "Trace Path"])[key]
    #     )
    #     base_val = df.set_index(["Cache Size", "Trace Path"]).index.map(base)
    #     df[f"Abs. D. {key}"] = df[key] - base_val
    #     df[f"Rel. D. {key}"] = (df[key] - base_val) / base_val


def PaperMeasurement(df: pd.DataFrame):
    fifo_miss_ratio = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["FIFO"]
        .set_index(["Cache Size", "Trace Path"])["Miss Ratio"]
    )
    df["Relative Miss Ratio [FIFO]"] = df["Miss Ratio"] / fifo_miss_ratio
    df["Promotion Efficiency"] = (
        (fifo_miss_ratio - df["Miss Ratio"]) * df["Request"] / df["Reinserted"]
    )

    lru_promos = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["LRU"]
        .set_index(["Cache Size", "Trace Path"])["Reinserted"]
    )
    df["Relative Promotion [LRU]"] = df["Reinserted"] / lru_promos

    lru_miss = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["LRU"]
        .set_index(["Cache Size", "Trace Path"])["Miss Ratio"]
    )
    df["Relative Miss Ratio [LRU]"] = df["Miss Ratio"] / lru_miss

    base_clock_condition = (df["Algorithm"] == "FR") & (df["Bit"] == 1)

    base_clock_promos = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.loc[base_clock_condition].set_index(["Cache Size", "Trace Path"])[
            "Reinserted"
        ]
    )
    df["Relative Promotion [Base FR]"] = df["Reinserted"] / base_clock_promos

    base_clock_miss = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.loc[base_clock_condition].set_index(["Cache Size", "Trace Path"])[
            "Miss Ratio"
        ]
    )
    df["Relative Miss Ratio [Base FR]"] = df["Miss Ratio"] / base_clock_miss

    bit_clock_promos = df.set_index(["Cache Size", "Trace Path", "Bit"]).index.map(
        df.loc[df["Algorithm"] == "FR"].set_index(["Cache Size", "Trace Path", "Bit"])[
            "Reinserted"
        ]
    )
    df["Relative Promotion [Bit FR]"] = df["Reinserted"] / bit_clock_promos

    bit_clock_miss = df.set_index(["Cache Size", "Trace Path", "Bit"]).index.map(
        df.loc[df["Algorithm"] == "FR"].set_index(["Cache Size", "Trace Path", "Bit"])[
            "Miss Ratio"
        ]
    )
    df["Relative Miss Ratio [Bit FR]"] = df["Miss Ratio"] / bit_clock_miss


def ReadData():
    files = sorted(
        glob(os.path.join(DATA_PATH, "**", "*.json"), recursive=True), key=sort_key
    )
    dfs: list[pd.DataFrame] = []
    for name, key in ALGO.items():
        if isinstance(key, str):
            key_files = [f for f in files if key in extract_desc(f)[1]]
            dfs.append(GetResult(key_files, name))
        elif isinstance(key, tuple):
            key_files = [f for f in files if key[0] in extract_desc(f)[1]]
            dfs.append(GetResult(key_files, name, key[1]))

    return pd.concat(dfs)


def ProcessData(df: pd.DataFrame):
    df = df.sort_values(by="Trace Path")
    df = df.query("`Real Cache Size` >= 10")
    df = df.query("`Request` >= 1_000_000")
    AdditionalProcessing(df)
    PaperMeasurement(df)
    return df


def PrintAGE(df: pd.DataFrame, writer: DocsWriter):
    writer.Write("# AGE")
    age = df.query("`Algorithm` == 'AGE'")
    clock = df.query("`Algorithm` in @BASE_ALGO and Bit == 1")
    data = pd.concat([age, clock], ignore_index=True)

    data["Scale"] = data["Scale"].astype(str)
    data.loc[data["Algorithm"].isin(BASE_ALGO), "Scale"] = data.loc[
        data["Algorithm"].isin(BASE_ALGO), "Algorithm"
    ]
    data = data.reset_index()
    for key in MEASUREMENTS:
        title = f"{key}"
        writer.Write(f"## {title}")
        fig = plt_box(
            data,
            y=key,
            x="Scale",
            hue="Algorithm",
            title=title,
            dodge=True,
            palette=PALETTE,
            showmeans=True,
        )
        writer.Write(fig)


def PrintPaperFigures(df: pd.DataFrame, writer: DocsWriter):
    palette = {
        "Prob": "lightblue",
        "Batch": "lightblue",
        "Delay": "lightblue",
        "FR": "lightblue",
        "D-FR": "lightgreen",
        "AGE": "lightgreen",
    }
    writer.Write("# FOR PAPER")

    age = df.query("`Algorithm` == 'AGE'")
    dclock = df.query("`Algorithm` == 'D-FR'")
    clock = df.query("Algorithm == 'FR'")
    other = df.query("Algorithm in ['Prob','Batch','Delay']")

    measurements = {
        "Relative Miss Ratio [LRU]": "Miss ratio relative to LRU",
        "Relative Promotion [LRU]": "Promotions relative to LRU",
        "Relative Miss Ratio [Base FR]": "Miss ratio relative to FR",
        "Relative Promotion [Base FR]": "Promotions relative to FR",
        "Relative Miss Ratio [Bit FR]": "Miss ratio relative to FR (same bit)",
        "Relative Promotion [Bit FR]": "Promotions relative to FR (same bit)",
        "Promotion Efficiency": "Promotion efficiency",
    }.items()
    data = pd.concat(
        [
            age.query("Scale == 0.5"),
            dclock.query("`Delay Ratio` == 0.05 and Bit == 1"),
            clock.query("Bit == 1"),
            other,
        ],
        ignore_index=True,
    )
    data = data.reset_index()
    for key, val in measurements:
        writer.Write(f"## {key}")
        fig = plt_box(
            data,
            y=key,
            y_label=val,
            x="Algorithm",
            x_label="Technique",
            hue="Algorithm",
            dodge=False,
            palette=palette,
            tick_step=0.2
            if "Relative Promotion" in key
            else 0.015
            if "Miss" in key
            else None,
            order=["Prob", "Batch", "Delay", "FR", "AGE", "D-FR"],
            x_size=12,
            width=0.7,
            output_pdf=f"../docs/dclock_{key.replace(' ', '_').lower()}.pdf",
        )
        writer.Write(fig)
    for aggfunc in ["mean", "median"]:
        writer.Write(f"## dclock {aggfunc} heatmap")
        for key, val in measurements:
            writer.Write(f"### {key}")
            fig = heat_map(
                dclock,
                x="Bit",
                y="Delay Ratio",
                title=val,
                values=key,
                aggfunc=aggfunc,
            )
            writer.Write(fig)
    for dr in dclock["Delay Ratio"].unique():
        print("dr: ", dr)
        data = pd.concat(
            [
                dclock.query("`Delay Ratio` == @dr"),
                clock,
            ],
            ignore_index=True,
        )
        print(data)
        writer.Write(f"## Delay Ratio : {dr}")
        for key, val in measurements:
            writer.Write(f"### {key}")
            fig = plt_box(
                data,
                y=key,
                y_label=val,
                x="Bit",
                hue="Algorithm",
                dodge=True,
                palette=palette,
                tick_step=0.015 if "Miss" in key else None,
                x_size=12,
                width=0.7,
                show_legend=True,
                # output_pdf=f"../docs/dclock_{key.replace(' ', '_').lower()}.pdf",
            )
            writer.Write(fig)


def GenerateSite(
    title: str,
    df: pd.DataFrame,
):
    if df.empty:
        return

    html_path = os.path.join(
        OUTPUT_PATH,
        f"{title}.html",
    )
    writer = DocsWriter(html_path=html_path, md_path=None)
    PrintPaperFigures(df, writer)
    writer.Flush()
    print("Finished generating " + title)


def main():
    age_results = age.ReadData()
    other_results = other.ReadData()
    df = ReadData()
    df = pd.concat([df, age_results, other_results], ignore_index=True)
    df = df.round(4)
    df = df.query("`Ignore Obj Size` == 1")
    df = ProcessData(df)
    GenerateSite("index", df)


if __name__ == "__main__":
    main()
