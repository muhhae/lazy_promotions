from glob import glob
import re
import os
import pandas as pd

pattern = re.compile(
    r".*?\s+(?P<algo>[A-Za-z0-9_]+)-(?P<age>[\d\.]+)\s+cache size\s+(?P<cache_size>\d+),\s+"
    r"(?P<requests>\d+)\s+req,\s+miss ratio\s+(?P<miss_ratio>[\d\.]+),\s+"
    r"throughput\s+(?P<throughput>[\d\.]+)\s+MQPS,\s+promotion\s+(?P<promotion>\d+)"
)


def parse_line(line: str, filename: str, cache_size: float):
    algorithms = {
        "lpFIFO_batch": "Batch",
        "LRU_delay": "Delay",
        "lpLRU_prob": "Prob",
    }
    match = pattern.search(line)
    if match:
        d = match.groupdict()
        return {
            "Algorithm": algorithms[d["algo"]],
            "Scale": float(d["age"]),
            "Real Cache Size": int(d["cache_size"]),
            "Request": int(d["requests"]),
            "Miss Ratio": float(d["miss_ratio"]),
            "Reinserted": int(d["promotion"]),
            "Trace": filename[filename.rfind("/") + 1 :],
            "Trace Path": filename,
            "Cache Size": 0.01,
            "Ignore Obj Size": 1,
            "Inserted": None,
            "Hit": None,
            "P": None,
            "Precision": None,
            "Ghost Size": None,
            "Threshold": None,
            "JSON File": None,
        }
    return None


def ReadData():
    with open("./datasets.txt", "r") as f:
        paths = f.readlines()
    paths = [p.strip() for p in paths]
    datasets = {
        p[p.rfind("/") + 1 :]: re.sub(r"\.oracleGeneral\S*", "", p) for p in paths
    }
    rows = []
    outputs = sorted(glob("../new_results/**/*"))
    for file in outputs:
        if os.path.isdir(file):
            continue
        with open(file, "r") as f:
            filename = file[: file.find(".cachesim")]
            for _, line in enumerate(f):
                key = filename[filename.rfind("/") + 1 :]
                if key not in datasets:
                    print(key, "doesn't exist!")
                    continue
                row = parse_line(line, datasets[key], 0.1)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)
