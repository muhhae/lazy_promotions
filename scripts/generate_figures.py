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

from matplotlib_wrapper import plt_box

OUTPUT_PATH: Final[str] = "../docs/"
DATA_PATH: Final[str] = "../simulation_results/hashed/"

MEASUREMENTS = [
    "Relative Miss Ratio [LRU]",
    "Relative Promotion [LRU]",
    "Relative Miss Ratio [FR]",
    "Relative Promotion [FR]",
    "Abs. D. Miss Ratio",
    # "Relative Miss Ratio [FIFO]",
    # "Promotion Efficiency",
]

ALGO: dict[str, str | tuple[str, int]] = {
    "FIFO": "fifo",
    # "SxFIFO": "sxfifo",
    "LRU": "lru",
    # "Q-Clock": "q-clock",
    # "Q-AUTO": "qauto",
    # "Q2-AUTO": "q2auto",
    # "Q3-AUTO": "q3auto",
    # "T-AUTO": "tauto",
    # "T2-AUTO": "t2auto",
    # "T3-AUTO": "t3auto",
    # "T4-AUTO": "t4auto",
    # "T5-AUTO": "t5auto",
    # "T6-AUTO": "t6auto",
    # "T7-AUTO": "t7auto",
    # "TIME-AUTO": "time-auto",
    # "TIME2-AUTO": "time2-auto",
    # "TIME3-AUTO": "time3-auto",
    # "QTime-Clock": "qtime-clock",
    "S3FClock": "s3fclock",
    "Gated Clock": "s3fclock-sequential",
    "D-CLOCK": "cm-clock",
    # "QAND-Clock": "qand-clock",
    # "QAND-Clock-v2": "qand-clock-v2",
    # "QOR-Clock": "qor-clock",
    "FR": ("offline-clock", 0),
    "Offline FR": ("offline-clock", 1),
    # "Offline Q-FR": ("offline-q-clock", 1),
    # "Offline QTime-Clock": ("offline-qtime-clock", 1),
}
ALGO_ORDERS = [
    "FIFO",
    "LRU",
    "FR",
    "Offline FR",
    "Q-Clock",
    "QAND-Clock",
    "QTime-Clock",
]
BASE_ALGO = [
    # "FIFO",
    "FR",
    # "Offline FR",
]
PARAMETERIZED_ALGO = [
    "QTime-Clock",
    "Q-Clock",
    "QAND-Clock",
    "QAND-Clock-v2",
    # "QOR-Clock",
]
GHOST_ALGO = [
    "T-AUTO",
    "T2-AUTO",
    "T3-AUTO",
    "T4-AUTO",
    "T5-AUTO",
    "T6-AUTO",
    "T7-AUTO",
    "TIME-AUTO",
    "TIME2-AUTO",
    "TIME3-AUTO",
]
PARAMETER_AUTO = [
    "Q-AUTO",
    "Q2-AUTO",
    "Q3-AUTO",
]
SFIFO = [
    "SxFIFO",
]
S3FClock = [
    # "S3FClock",
    "Gated Clock",
    "D-CLOCK",
]

PALETTE = ["lightblue", "lightgreen", "lightpink", "purple", "gray"]
PALETTE = "pastel"
Ps = [i / 8 for i in range(1, 8)]
# Ps = [0.0625] + Ps
Ps = [str(i) for i in Ps]


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
            category_orders={"Algorithm": ALGO_ORDERS},
        )
        writer.WriteFig(fig)


def WriteAggregateP(
    writer: DocsWriter,
    df: pd.DataFrame,
    desc: str = "",
):
    writer.Write(f"## Aggregate Results {desc}")
    for key in MEASUREMENTS:
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


def AdditionalProcessing(df: pd.DataFrame, compared_algo="FR"):
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

    clock_promos = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["FR"]
        .set_index(["Cache Size", "Trace Path"])["Reinserted"]
    )
    df["Relative Promotion [FR]"] = df["Reinserted"] / clock_promos

    clock_miss = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["FR"]
        .set_index(["Cache Size", "Trace Path"])["Miss Ratio"]
    )
    df["Relative Miss Ratio [FR]"] = df["Miss Ratio"] / clock_miss


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
    print("Trace By Path: ", len(df["Trace Path"].unique()))
    print("Trace By Name: ", len(df["Trace"].unique()))

    df = df.query("`Real Cache Size` >= 10")
    print("Excluding cache_size < 10")
    print("Trace By Path: ", len(df["Trace Path"].unique()))
    print("Trace By Name: ", len(df["Trace"].unique()))

    df = df.query("`Request` >= 1_000_000")
    print("Excluding request < 1_000_000")
    print("Trace By Path: ", len(df["Trace Path"].unique()))
    print("Trace By Name: ", len(df["Trace"].unique()))

    clock = df.query("Algorithm == 'FR'")
    print("[FR] Trace By Path: ", len(clock["Trace Path"].unique()))
    print("[FR] Trace By Name: ", len(clock["Trace"].unique()))

    AdditionalProcessing(df)
    PaperMeasurement(df)

    df = df.sort_values(by="P")
    df["P"] = df["P"].astype(str)
    df.loc[~df["Algorithm"].isin(PARAMETERIZED_ALGO), "P"] = df.loc[
        ~df["Algorithm"].isin(PARAMETERIZED_ALGO), "Algorithm"
    ]
    print(df["Cache Size"].unique())
    print(df["Algorithm"].unique())
    # df.to_pickle("df.pkl")
    # print(df["P"].unique())
    return df


def PrintAGE(df: pd.DataFrame, writer: DocsWriter):
    writer.Write("# AGE")
    data = df.query("`Algorithm` == 'AGE' or `Algorithm` in @BASE_ALGO")

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
            # tick_step=0.005 if "Miss" in key else None,
            showmeans=True,
        )
        writer.Write(fig)


def PrintPaperFigures(df: pd.DataFrame, writer: DocsWriter):
    palette = {
        "Prob": "lightblue",
        "Batch": "lightblue",
        "Delay": "lightblue",
        "FR": "lightblue",
        "D-CLOCK": "lightgreen",
        "AGE": "lightgreen",
    }
    __import__("pandasgui").show(df)
    writer.Write("# FOR PAPER")
    df = df[np.isfinite(df["Promotion Efficiency"])]
    print(df.groupby("Algorithm")["Promotion Efficiency"].agg(["mean", "median"]))
    age = df.query("`Algorithm` == 'AGE'")
    dclock = df.query("`Algorithm` == 'D-CLOCK'")
    base = df.query("Algorithm in @BASE_ALGO or Algorithm in ['Prob','Batch','Delay']")
    measurements = {
        "Relative Miss Ratio [LRU]": "Miss ratio relative to LRU",
        "Relative Promotion [LRU]": "Promotions relative to LRU",
        "Promotion Efficiency": "Promotion efficiency",
        "Relative Miss Ratio [FR]": "Miss ratio relative to FR",
        "Relative Promotion [FR]": "Promotions relative to FR",
    }.items()

    data = pd.concat(
        [age.query("Scale == 0.5"), dclock.query("`Hand Position` == 0.05"), base],
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
            tick_step=0.2 if "Promotion" in key else 0.01 if "Miss" in key else None,
            order=["Prob", "Batch", "Delay", "FR", "AGE", "D-CLOCK"],
            x_size=12,
        )
        writer.Write(fig)

    for key, val in measurements:
        writer.Write(f"## {key}")
        fig = plt_box(
            dclock.query("`Hand Position` >= 0.01 and `Hand Position` <= 0.5"),
            y=key,
            y_label=val,
            x="Hand Position",
            x_label="Delay Ratio",
            hue="Algorithm",
            palette=palette,
            tick_step=0.2 if "Promotion" in key else 0.01 if "Miss" in key else None,
            x_size=12,
        )
        writer.Write(fig)

    dclock_base = pd.concat(
        [
            dclock.query("`Hand Position` >= 0.01 and `Hand Position` <= 0.5"),
            base,
        ],
        ignore_index=True,
    )
    dclock_base.loc[dclock_base["Algorithm"].isin(BASE_ALGO), "Hand Position"] = (
        dclock_base.loc[dclock_base["Algorithm"].isin(BASE_ALGO), "Algorithm"]
    )
    for key, val in measurements:
        writer.Write(f"## {key}")
        fig = plt_box(
            dclock_base,
            y=key,
            y_label=val,
            x="Hand Position",
            x_label="Delay Ratio",
            hue="Algorithm",
            palette=palette,
            tick_step=0.2 if "Promotion" in key else 0.01 if "Miss" in key else None,
            x_size=12,
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
    print(age_results)
    other_results = other.ReadData()
    print(other_results)
    traces = age_results["Trace Path"].unique()
    print(len(traces))
    traces = other_results["Trace Path"].unique()
    print(len(traces))
    df = ReadData()
    df = df.query("`Trace Path` in @traces")
    print(len(df["Trace Path"].unique()))
    df = pd.concat([df, age_results, other_results], ignore_index=True)
    df = df.round(4)
    print(len(df["Trace Path"].unique()))
    df = df.query("`Ignore Obj Size` == 1")
    df = ProcessData(df)
    print(df["Algorithm"].unique())
    GenerateSite("index", df)


if __name__ == "__main__":
    main()
