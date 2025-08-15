import os
from glob import glob
from pathlib import Path
from typing import Final

import pandas as pd

from common import extract_desc, sort_key
from outputs_parser import GetResult
from docs_writer import DocsWriter
from plotly_wrapper import Box, Scatter

OUTPUT_PATH: Final[str] = "../docs/"
DATA_PATH: Final[str] = "../paper/log/"
ALGO_ORDERS = [
    "FIFO",
    "LRU",
    "CLOCK",
    "Offline CLOCK",
    "Q-Clock",
    "QTime-Clock",
]


def WriteAggregate(
    writer: DocsWriter,
    df: pd.DataFrame,
    desc: str = "",
):
    writer.Write(f"## Aggregate Results for {desc}")
    for key in [
        "[PAPER] Relative Miss Ratio",
        "[PAPER] Relative Promotion",
        "[PAPER] Promotion Efficiency",
    ]:
        title = f"{key} for {desc}"
        writer.Write(f"### {title}")
        fig = Box(
            df,
            y=key,
            x="Algorithm",
            title=title,
            category_orders={"Algorithm": ALGO_ORDERS},
        )
        writer.WriteFig(fig)

    for key in ["Hit", "Miss Ratio", "Reinserted"]:
        title = f"Relative Delta {key} for {desc}"
        writer.Write(f"### {title}")
        fig = Box(
            df.query("Algorithm != 'LRU'") if key == "Reinserted" else df,
            y=f"Rel. D. {key}",
            x="Algorithm",
            title=title,
            category_orders={"Algorithm": ALGO_ORDERS},
        )
        writer.WriteFig(fig)
    for key in ["Hit", "Miss Ratio", "Reinserted"]:
        title = f"Absolute Delta {key}"
        writer.Write(f"### {title}")
        fig = Box(
            df.query("Algorithm != 'LRU'") if key == "Reinserted" else df,
            y=f"Abs. D. {key}",
            x="Algorithm",
            title=title,
            category_orders={"Algorithm": ALGO_ORDERS},
        )
        writer.WriteFig(fig)


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
    for p in sorted(df["P"].unique()):
        data = df.query("`P` == @p or `Algorithm` not in ['QTime-Clock', 'Q-Clock']")
        if data.empty:
            continue

        current_title = f"P = {p}"
        writer.Write(f"# {current_title}")
        WriteAggregate(writer, data, current_title)
        WriteIndividual(writer, data)

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
    lru_promos = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["LRU"]
        .set_index(["Cache Size", "Trace Path"])["Reinserted"]
    )
    fifo_miss = df.set_index(["Cache Size", "Trace Path"]).index.map(
        df.set_index("Algorithm")
        .loc["FIFO"]
        .set_index(["Cache Size", "Trace Path"])["Miss"]
    )
    df["[PAPER] Relative Miss Ratio"] = df["Miss"] / fifo_miss
    df["[PAPER] Relative Promotion"] = df["Reinserted"] / lru_promos
    df["[PAPER] Promotion Efficiency"] = (fifo_miss - df["Miss"]) / df["Reinserted"]


def main():
    files = sorted(glob(os.path.join(DATA_PATH, "*.json")), key=sort_key)

    alg: dict[str, str | tuple[str, int]] = {
        "FIFO": "fifo",
        "LRU": "lru",
        "Q-Clock": "q-clock",
        "QTime-Clock": "qtime-clock",
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

    df = pd.concat(dfs)
    df = df.sort_values(by="Trace Path")
    AdditionalProcessing(df)
    PaperMeasurement(df)
    GenerateSite("index", df)


if __name__ == "__main__":
    main()
