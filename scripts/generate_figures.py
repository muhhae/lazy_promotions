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
    fifo_miss = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["FIFO"]
        .set_index(["Cache Size", "Trace Path"])["Miss Ratio"]
    )
    df["Relative Miss Ratio [FIFO]"] = df["Miss Ratio"] / fifo_miss
    df["Promotion Efficiency"] = (
        (fifo_miss - df["Miss Ratio"]) / df["Reinserted"] * df["Request"]
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


def ProcessCustomAlgo(df: pd.DataFrame, key):
    df = df.query("Algorithm in @S3FClock")
    grouped = df.groupby(["Cache Size", "Trace Path", "Algorithm"], group_keys=False)

    def pick_with_larger_P(sub_df, col="Relative Miss Ratio [FR]", mode="min"):
        sub_df = sub_df.dropna(subset=[col])
        if sub_df.empty:
            return None
        if mode == "min":
            target_val = sub_df[col].min()
        elif mode == "max":
            target_val = sub_df[col].max()
        else:
            raise ValueError("mode must be 'min' or 'max'")

        candidates = sub_df[sub_df[col] == target_val]
        chosen_P = candidates.loc[candidates["Hand Position"].idxmax(), key]
        return chosen_P

    lowest = grouped.apply(lambda g: pick_with_larger_P(g, mode="min"))
    lowest = pd.Series(lowest, name="lowest miss ratio")
    result = pd.concat(
        [
            lowest,
        ],
        axis=1,
    ).reset_index()

    return result


def PrintParameterDistribution(df: pd.DataFrame, writer: DocsWriter):
    writer.Write("# Parameter Distribution")
    for key in [
        "Hand Position",
        "Relative Miss Ratio [FR]",
        "Relative Promotion [FR]",
    ]:
        q = ProcessCustomAlgo(df, key)
        q_melt = q.melt(
            id_vars=["Cache Size", "Trace Path", "Algorithm"],
            var_name="X",
            value_name=key,
        )
        q_melt = q_melt.query(f"`{key}` == `{key}`")
        q_melt[key] = q_melt[key].astype(float)
        title = f"{key} Distribution"
        fig = plt_box(
            q_melt,
            y=key,
            x="X",
            hue="Algorithm",
            title=title,
            dodge=True,
            palette=PALETTE,
            # tick_step=0.125 if key == "P" else None,
        )
        writer.Write(fig)


def PrintS3FClock(df: pd.DataFrame, writer: DocsWriter):
    writer.Write("# S3FClock")
    data = df.query("`Algorithm` in @BASE_ALGO or `Algorithm` in @S3FClock")
    for X in ["Hand Position"]:
        data[X] = data[X].astype(str)
        data.loc[data["Algorithm"].isin(BASE_ALGO), X] = data.loc[
            data["Algorithm"].isin(BASE_ALGO), "Algorithm"
        ]
    data = data.reset_index()
    for key in MEASUREMENTS:
        title = f"{key}"
        writer.Write(f"## {key}")
        fig = plt_box(
            data,
            y=key,
            x="Hand Position",
            hue="Algorithm",
            title=title,
            dodge=True,
            palette=PALETTE,
            legend_font_size=18,
            tick_step=0.01 if "Miss" in key else None,
        )
        writer.Write(fig)


def PrintOverall(df: pd.DataFrame, writer: DocsWriter):
    writer.Write("# OVERALL")
    data = df.query(
        "`Algorithm` in @BASE_ALGO or (`Algorithm` == 'AGE' and Scale == 0.5) or (`Algorithm` in @S3FClock and `Hand Position` <= 0.2 and `Hand Position` > 0)"
    )
    for X in ["Hand Position"]:
        data[X] = data[X].astype(str)
        data.loc[~data["Algorithm"].isin(S3FClock), X] = data.loc[
            ~data["Algorithm"].isin(S3FClock), "Algorithm"
        ]
    data = data.reset_index()
    for key in MEASUREMENTS:
        title = f"{key}"
        writer.Write(f"## {key}")
        fig = plt_box(
            data,
            y=key,
            x="Hand Position",
            hue="Algorithm",
            title=title,
            dodge=True,
            palette=PALETTE,
            # order=["Gated Clock", "AGE", "QAND-Clock", "FR"],
            # tick_step=0.005 if "Miss" in key else None,
            legend_font_size=18,
        )
        writer.Write(fig)


def PrintOverallAdaptive(df: pd.DataFrame, writer: DocsWriter):
    writer.Write("# OVERALL AUTO")
    ghost_algo = ["T3-AUTO", "T5-AUTO", "T6-AUTO", "T7-AUTO"]
    parameter_auto = ["Q2-AUTO", "Q3-AUTO"]
    parameterized = ["QAND-Clock", "QAND-Clock-v2"]
    data = df.query(
        "(`Algorithm` in @parameter_auto and `Precision` == 16) or `Algorithm` in @ghost_algo or `Algorithm` in @BASE_ALGO or (`Algorithm` in @parameterized and `P` == '0.375') or (`Algorithm` == 'AGE' and Scale == '0.5')"
    )

    data = data.reset_index()
    data["Ghost Size"] = data["Ghost Size"].astype(str)
    data.loc[~data["Algorithm"].isin(GHOST_ALGO + PARAMETER_AUTO), "Ghost Size"] = (
        data.loc[~data["Algorithm"].isin(GHOST_ALGO + PARAMETER_AUTO), "Algorithm"]
    )

    for key in MEASUREMENTS:
        title = f"{key}"
        writer.Write(f"## {key}")
        fig = plt_box(
            data,
            y=key,
            x="Ghost Size",
            hue="Algorithm",
            title=title,
            dodge=True,
            palette=PALETTE,
            tick_step=0.005 if "Miss" in key else None,
            legend_font_size=18,
        )
        writer.Write(fig)


def PrintAdaptive(df: pd.DataFrame, writer: DocsWriter):
    writer.Write("# AUTO")
    data = df.query("`Algorithm` in @GHOST_ALGO or `Algorithm` in @BASE_ALGO")

    data = data.reset_index()
    for a in GHOST_ALGO:
        print(
            f"{a}: ",
            len(data.query("Algorithm == @a")["Trace Path"].unique()),
        )

    data["Ghost Size"] = data["Ghost Size"].astype(str)
    data.loc[~data["Algorithm"].isin(GHOST_ALGO), "Ghost Size"] = data.loc[
        ~data["Algorithm"].isin(GHOST_ALGO), "Algorithm"
    ]

    for key in MEASUREMENTS:
        title = f"{key}"
        writer.Write(f"## {key}")
        fig = plt_box(
            data,
            y=key,
            x="Ghost Size",
            hue="Algorithm",
            title=title,
            dodge=True,
            palette=PALETTE,
            # tick_step=0.005 if "Miss" in key else None,
        )
        writer.Write(fig)
    for a in GHOST_ALGO:
        tmp = data.query("`Algorithm` == @a or `Algorithm` in @BASE_ALGO")
        writer.Write(f"# {a}")
        for key in MEASUREMENTS:
            title = f"{key}"
            writer.Write(f"## {key}")
            fig = plt_box(
                tmp,
                y=key,
                x="Ghost Size",
                hue="Algorithm",
                title=title,
                dodge=False,
                palette=PALETTE,
                # tick_step=0.005 if "Miss" in key else None,
            )
            writer.Write(fig)


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


def PrintQ(df: pd.DataFrame, writer: DocsWriter):
    writer.Write("# Q-AUTO")
    data = df.query("`Algorithm` in @PARAMETER_AUTO or `Algorithm` in @BASE_ALGO")
    data = data.query("`Precision` < 32 or `Algorithm` in @BASE_ALGO")

    data["Precision"] = data["Precision"].astype(str)
    data["Ghost Size"] = data["Ghost Size"].astype(str)

    data.loc[~data["Algorithm"].isin(PARAMETER_AUTO), "Precision"] = data.loc[
        ~data["Algorithm"].isin(PARAMETER_AUTO), "Algorithm"
    ]
    data.loc[~data["Algorithm"].isin(PARAMETER_AUTO), "Ghost Size"] = data.loc[
        ~data["Algorithm"].isin(PARAMETER_AUTO), "Algorithm"
    ]

    data = data.reset_index()
    for a in PARAMETER_AUTO:
        print(
            f"{a}: ",
            len(data.query("Algorithm == @a")["Trace Path"].unique()),
        )
    for a in PARAMETER_AUTO:
        for key in MEASUREMENTS:
            title = f"[{a}] {key}"
            writer.Write(f"## {title}")
            fig = plt_box(
                data.query("`Algorithm` == @a or `Algorithm` in @BASE_ALGO"),
                y=key,
                x="Precision",
                hue="Ghost Size",
                title=title,
                dodge=True,
                palette=PALETTE,
                # tick_step=0.005 if "Miss" in key else None,
                showmeans=False,
            )
            writer.Write(fig)


def PrintPaperFigures(df: pd.DataFrame, writer: DocsWriter):
    writer.Write("# FOR PAPER")
    age = df.query("`Algorithm` == 'AGE'")
    dclock = df.query("`Algorithm` == 'D-CLOCK'")
    base = df.query("Algorithm in @BASE_ALGO")

    measurements = {
        "Relative Miss Ratio [LRU]": "Miss ratio relative to LRU",
        "Relative Promotion [LRU]": "Promotions relative to LRU",
        "Promotion Efficiency": "Promotion efficiency",
        "Relative Miss Ratio [FR]": "Miss ratio relative to FR",
        "Relative Promotion [FR]": "Promotions relative to FR",
    }.items()

    print("FR: ", len(base["Trace Path"].unique()))
    print("AGE: ", len(age["Trace Path"].unique()))
    print("DCLOCk: ", len(dclock["Trace Path"].unique()))

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
            palette=PALETTE,
            tick_step=0.2 if "Promotion" in key else 0.01 if "Miss" in key else None,
            order=["D-CLOCK", "AGE", "FR"],
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
            palette=PALETTE,
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
            palette=["lightblue", "lightgreen"],
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
    writer.Write("# Overall P")
    df = df.query(
        "Algorithm in @S3FClock or Algorithm in @SFIFO or Algorithm == 'AGE' or P in @Ps or Algorithm in @BASE_ALGO or Algorithm in @PARAMETER_AUTO or Algorithm in @GHOST_ALGO"
    )
    # WriteAggregatePLT(writer, df.query("P in @Ps or Algorithm in @BASE_ALGO"))
    # for Algo in PARAMETERIZED_ALGO:
    #     writer.Write(f"# {Algo} compared to base_algo")
    #     WriteAggregatePLT(
    #         writer,
    #         df.query("`Algorithm` == @Algo or `Algorithm` in @BASE_ALGO"),
    #         f"# {Algo} compared to base_algo",
    #     )
    #     writer.Write(f"# {Algo} alone")
    #     WriteAggregatePLT(
    #         writer,
    #         df.query("`Algorithm` == @Algo"),
    #         f"# {Algo} alone",
    #     )
    # PrintSxFIFO(df, writer)
    PrintS3FClock(df, writer)
    # PrintAGE(df, writer)
    # PrintQ(df, writer)
    # PrintAdaptive(df, writer)
    # PrintOverallAdaptive(df, writer)
    # PrintOverall(df, writer)
    # PrintParameterDistribution(df, writer)
    PrintPaperFigures(df, writer)
    writer.Flush()
    print("Finished generating " + title)


def main():
    age_results = age.ReadData()
    print(age_results)
    traces = age_results["Trace Path"].unique()
    print(len(traces))
    df = ReadData()
    df = df.query("`Trace Path` in @traces")
    print(len(df["Trace Path"].unique()))
    df = pd.concat([df, age_results], ignore_index=True)
    print(len(df["Trace Path"].unique()))
    # exit(0)
    df = df.query("`Ignore Obj Size` == 1")
    df = ProcessData(df)
    print(df["Algorithm"].unique())
    GenerateSite("index", df)


if __name__ == "__main__":
    main()
