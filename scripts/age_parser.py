from glob import glob
from pprint import pprint
import re
import pandas as pd

pattern = re.compile(
    r"AGE-(?P<age>[\d\.]+)\s+cache size\s+(?P<cache_size>\d+),\s+"
    r"(?P<requests>\d+)\s+req,\s+miss ratio\s+(?P<miss_ratio>[\d\.]+),\s+"
    r"byte miss ratio\s+(?P<byte_miss_ratio>[\d\.]+)\s+(?P<reinserted>\d+)"
)


def parse_line(line: str, filename: str, cache_size: float):
    match = pattern.search(line)
    if match:
        d = match.groupdict()
        return {
            "Algorithm": "AGE",
            "Scale": float(d["age"]),
            "Real Cache Size": int(d["cache_size"]),
            "Request": int(d["requests"]),
            "Miss Ratio": float(d["miss_ratio"]),
            "Reinserted": int(d["reinserted"]),
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
    print(len(datasets))
    rows = []
    outputs = sorted(glob("../age_results/*"))
    for file in outputs:
        with open(file, "r") as f:
            for i, line in enumerate(f):
                if i % 3 != 1:
                    continue
                key = file[file.rfind("/") + 1 :]
                if key not in datasets:
                    print(key, "doesn't exist!")
                    continue
                row = parse_line(line, datasets[key], 0.1)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)
