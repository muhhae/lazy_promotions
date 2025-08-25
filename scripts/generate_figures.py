import re
import io
import plotly.express as px
import plotly.graph_objects as go
import os
from glob import glob
from pathlib import Path
import pprint
from typing import Final
import numpy as np
import seaborn as sns

import pandas as pd
from common import extract_desc, sort_key
from docs_writer import DocsWriter
from outputs_parser import GetResult
from plotly_wrapper import Box, Scatter, Box_2
import matplotlib.pyplot as plt

from matplotlib_wrapper import plt_box

OUTPUT_PATH: Final[str] = "../docs/"
DATA_PATH: Final[str] = "../simulation_results/hashed/"

ALGO_ORDERS = [
    "FIFO",
    "LRU",
    "CLOCK",
    "Offline CLOCK",
    "Q-Clock",
    "QAND-Clock",
    "QTime-Clock",
]
BASE_ALGO = [
    # "FIFO",
    "CLOCK",
    # "Offline CLOCK",
]
CUSTOM_ALGO = [
    "QTime-Clock",
    "Q-Clock",
    "QAND-Clock",
    # "QOR-Clock",
]
PALETTE = ["lightblue", "lightgreen", "lightpink", "purple", "gray"]
# PALETTE = "pastel"
Ps = [i / 8 for i in range(2, 6)]
# Ps = [0.0625] + Ps
Ps = [str(i) for i in Ps]


def WriteIndividual(
    writer: DocsWriter,
    df: pd.DataFrame,
    title: str = "",
):
    writer.Write("## Individual Results")
    for ign_obj_size in df["Ignore Obj Size"].unique():
        writer.Write(f"### Ignore Object Size = {'True' if ign_obj_size else 'False'}")
        obj_size_filtered = df.query("`Ignore Obj Size` == @ign_obj_size")
        for trace_group in sorted(obj_size_filtered["Trace Group"].unique()):
            writer.Write(f"#### {trace_group}")
            group_filtered = obj_size_filtered.query("`Trace Group` == @trace_group")
            for trace in sorted(group_filtered["Trace Path"].unique()):
                writer.Write(f"##### {trace}")
                trace_filtered = group_filtered.query("`Trace Path` == @trace")
                for cache_size in trace_filtered["Cache Size"].unique():
                    writer.Write(f"###### Cache Size = {cache_size}")
                    size_filtered = trace_filtered.query("`Cache Size` == @cache_size")
                    fig = Scatter(
                        size_filtered,
                        x="Reinserted",
                        y="Miss Ratio",
                        color="Algorithm",
                        title=f"{trace} {cache_size * 100}% {title}",
                        category_orders={"Algorithm": ALGO_ORDERS},
                        symbol="Algorithm",
                    )
                    writer.WriteFig(fig)
            print(f"{trace_group} generated!")
        print(f"{ign_obj_size} generated!")


def WriteAggregatePLT_P(
    writer: DocsWriter,
    df: pd.DataFrame,
    desc: str = "",
):
    df = df.reset_index()
    for key in [
        "Relative Miss Ratio [LRU]",
        "Relative Miss Ratio [FIFO]",
        "Relative Miss Ratio [CLOCK]",
        "Relative Promotion [LRU]",
        "Relative Promotion [CLOCK]",
        "Promotion Efficiency",
    ]:
        title = f"{desc}"
        writer.Write(f"## {key}")
        fig = plt_box(
            df,
            y=key,
            x="P",
            hue="Algorithm",
            title=title,
            dodge=False,
            palette=PALETTE,
            tick_step=0.01 if "Miss" in key else None,
        )
        writer.Write(fig)


def WriteAggregatePLT(
    writer: DocsWriter,
    df: pd.DataFrame,
    desc: str = "",
):
    df = df.reset_index()
    for key in [
        "Relative Miss Ratio [LRU]",
        "Relative Miss Ratio [FIFO]",
        "Relative Miss Ratio [CLOCK]",
        "Relative Promotion [LRU]",
        "Relative Promotion [CLOCK]",
        "Promotion Efficiency",
    ]:
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
    for key in [
        "Relative Miss Ratio [LRU]",
        "Relative Miss Ratio [FIFO]",
        "Relative Miss Ratio [CLOCK]",
        "Relative Promotion [LRU]",
        "Relative Promotion [CLOCK]",
        "Promotion Efficiency",
    ]:
        title = f"{key} for {desc}"
        writer.Write(f"### {key}")
        fig = Box(
            df,
            y=key,
            x="Algorithm",
            title=title,
            category_orders={"Algorithm": ALGO_ORDERS},
        )
        writer.WriteFig(fig)


def WriteAggregateP(
    writer: DocsWriter,
    df: pd.DataFrame,
    desc: str = "",
):
    writer.Write(f"## Aggregate Results {desc}")
    for key in [
        "Relative Miss Ratio [LRU]",
        "Relative Miss Ratio [FIFO]",
        "Relative Miss Ratio [CLOCK]",
        "Relative Promotion [LRU]",
        "Relative Promotion [CLOCK]",
        "Promotion Efficiency",
    ]:
        title = f"{key} {desc}"
        writer.Write(f"### {key}")
        fig = Box(
            df,
            y=key,
            x="P",
            color="Algorithm",
            title=title,
            category_orders={"Algorithm": ALGO_ORDERS},
        )
        fig.update_layout(showlegend=True)
        writer.WriteFig(fig)

    for key in ["Hit", "Miss Ratio", "Reinserted"]:
        title = f"Relative Delta {key} {desc}"
        writer.Write(f"### {key}")
        fig = Box(
            df,
            y=f"Rel. D. {key}",
            x="P",
            color="Algorithm",
            title=title,
            category_orders={"Algorithm": ALGO_ORDERS},
        )
        fig.update_layout(showlegend=True)
        writer.WriteFig(fig)
    for key in ["Hit", "Miss Ratio", "Reinserted"]:
        title = f"Absolute Delta {key}"
        writer.Write(f"### {key}")
        fig = Box(
            df,
            y=f"Abs. D. {key}",
            x="P",
            color="Algorithm",
            title=title,
            category_orders={"Algorithm": ALGO_ORDERS},
        )
        fig.update_layout(showlegend=True)
        writer.WriteFig(fig)


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
    writer.Write("# Overall P")
    df = df.query("P in @Ps or Algorithm in @BASE_ALGO or Algorithm == 'Q-Auto'")
    WriteAggregatePLT_P(writer, df)
    for Algo in CUSTOM_ALGO:
        writer.Write(f"# {Algo} compared to base_algo")
        WriteAggregatePLT_P(
            writer,
            df.query("`Algorithm` == @Algo or `Algorithm` in @BASE_ALGO"),
            f"# {Algo} compared to base_algo",
        )
        writer.Write(f"# {Algo} alone")
        WriteAggregatePLT_P(
            writer,
            df.query("`Algorithm` == @Algo"),
            f"# {Algo} alone",
        )
    Algo = "Q-Auto"
    writer.Write(f"# {Algo} threshold 0.1 compared to base_algo")
    data = df.query(
        "(`Algorithm` == @Algo and Threshold == 0.1) or `Algorithm` in @BASE_ALGO"
    )
    data = data.reset_index()

    for key in [
        "Relative Miss Ratio [LRU]",
        "Relative Miss Ratio [FIFO]",
        "Relative Miss Ratio [CLOCK]",
        "Relative Promotion [LRU]",
        "Relative Promotion [CLOCK]",
        "Promotion Efficiency",
    ]:
        title = f"{Algo}"
        writer.Write(f"## {key}")
        fig = plt_box(
            data,
            y=key,
            x="Precision",
            hue="Algorithm",
            title=title,
            dodge=True,
            palette=PALETTE,
            tick_step=0.01 if "Miss" in key else None,
        )
        writer.Write(fig)

    data = df.query(
        "(`Algorithm` == @Algo and Threshold == 0.01) or `Algorithm` in @BASE_ALGO"
    )
    data = data.reset_index()

    writer.Write(f"# {Algo} threshold 0.01 compared to base_algo")
    for key in [
        "Relative Miss Ratio [LRU]",
        "Relative Miss Ratio [FIFO]",
        "Relative Miss Ratio [CLOCK]",
        "Relative Promotion [LRU]",
        "Relative Promotion [CLOCK]",
        "Promotion Efficiency",
    ]:
        title = f"{Algo}"
        writer.Write(f"## {key}")
        fig = plt_box(
            data,
            y=key,
            x="Precision",
            hue="Algorithm",
            title=title,
            dodge=True,
            palette=PALETTE,
            tick_step=0.01 if "Miss" in key else None,
        )
        writer.Write(fig)

    # writer.Write("# Individual P")
    # for p in sorted(df["P"].unique()):
    #     data = df.query("`P` == @p or `Algorithm` not in @CUSTOM_ALGO")
    #     if data.empty:
    #         continue
    #     current_title = f"P = {p}"
    #     writer.Write(f"## {current_title}")
    #     WriteAggregatePLT(writer, data, current_title)

    writer.Flush()
    print("Finished generating " + title)


def AdditionalProcessing(df: pd.DataFrame, compared_algo="CLOCK"):
    df["Trace Group"] = df["Trace Path"].apply(lambda x: Path(x).parts[0])
    df["Miss"] = df["Request"] - df["Hit"]
    for key in ["Hit", "Miss Ratio", "Reinserted"]:
        base = (
            df.set_index("Algorithm")
            .loc[compared_algo]
            .set_index(["Cache Size", "Trace Path"])[key]
        )
        base_val = df.set_index(["Cache Size", "Trace Path"]).index.map(base)
        df[f"Abs. D. {key}"] = df[key] - base_val
        df[f"Rel. D. {key}"] = (df[key] - base_val) / base_val


def PaperMeasurement(df: pd.DataFrame):
    fifo_miss = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["FIFO"]
        .set_index(["Cache Size", "Trace Path"])["Miss"]
    )
    df["Relative Miss Ratio [FIFO]"] = df["Miss"] / fifo_miss
    df["Promotion Efficiency"] = (fifo_miss - df["Miss"]) / df["Reinserted"]

    lru_promos = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["LRU"]
        .set_index(["Cache Size", "Trace Path"])["Reinserted"]
    )
    df["Relative Promotion [LRU]"] = df["Reinserted"] / lru_promos

    lru_miss = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["LRU"]
        .set_index(["Cache Size", "Trace Path"])["Miss"]
    )
    df["Relative Miss Ratio [LRU]"] = df["Miss"] / lru_miss

    clock_promos = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["CLOCK"]
        .set_index(["Cache Size", "Trace Path"])["Reinserted"]
    )
    df["Relative Promotion [CLOCK]"] = df["Reinserted"] / clock_promos

    clock_miss = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["CLOCK"]
        .set_index(["Cache Size", "Trace Path"])["Miss"]
    )
    df["Relative Miss Ratio [CLOCK]"] = df["Miss"] / clock_miss


def ReadData():
    files = sorted(
        glob(os.path.join(DATA_PATH, "**", "*.json"), recursive=True), key=sort_key
    )

    alg: dict[str, str | tuple[str, int]] = {
        "FIFO": "fifo",
        "LRU": "lru",
        "Q-Clock": "q-clock",
        "Q-Auto": "qauto",
        "QTime-Clock": "qtime-clock",
        "QAND-Clock": "qand-clock",
        "QOR-Clock": "qor-clock",
        "CLOCK": ("offline-clock", 0),
        "Offline CLOCK": ("offline-clock", 1),
        # "Offline Q-CLOCK": ("offline-q-clock", 1),
        # "Offline QTime-Clock": ("offline-qtime-clock", 1),
    }

    dfs: list[pd.DataFrame] = []
    for name, key in alg.items():
        if isinstance(key, str):
            key_files = [f for f in files if key in extract_desc(f)[1]]
            dfs.append(GetResult(key_files, name))
        elif isinstance(key, tuple):
            key_files = [f for f in files if key[0] in extract_desc(f)[1]]
            dfs.append(GetResult(key_files, name, key[1]))

    return pd.concat(dfs)


def ProcessData(df: pd.DataFrame):
    df = df.sort_values(by="Trace Path")

    with open("./datasets.txt", "r") as f:
        paths = f.readlines()

    print(len(paths))
    paths = [re.sub(r"\.oracleGeneral\S*", "", line).strip() for line in paths]

    print("Trace By Path: ", len(df["Trace Path"].unique()))
    print("Trace By Name: ", len(df["Trace"].unique()))

    df = df.query("`Real Cache Size` > 10")
    print("Excluding cache_size < 10")
    print("Trace By Path: ", len(df["Trace Path"].unique()))
    print("Trace By Name: ", len(df["Trace"].unique()))

    df = df.query("`Request` > 1_000_000")
    print("Excluding request < 1_000_000")
    print("Trace By Path: ", len(df["Trace Path"].unique()))
    print("Trace By Name: ", len(df["Trace"].unique()))

    df = df.query("`Trace Path` in @paths")
    print("Excluding request not in paths")
    print("Trace By Path: ", len(df["Trace Path"].unique()))
    print("Trace By Name: ", len(df["Trace"].unique()))

    AdditionalProcessing(df)
    PaperMeasurement(df)
    df = df.sort_values(by="P")
    df["P"] = df["P"].astype(str)
    df.loc[~df["Algorithm"].isin(CUSTOM_ALGO), "P"] = df.loc[
        ~df["Algorithm"].isin(CUSTOM_ALGO), "Algorithm"
    ]
    print(df["Cache Size"].unique())
    df.to_pickle("df.pkl")
    print(df["P"].unique())
    return df


def main():
    df = ReadData()
    df = ProcessData(df)
    # __import__("pandasgui").show(df)
    # return
    GenerateSite("index", df)


if __name__ == "__main__":
    main()
