import json
import os
from pathlib import Path
from typing import Any, List, cast

from numpy import nan, trace
import pandas as pd
from common import extract_desc


def ProcessResultJSON(result: dict, file, algorithm):
    prefix, desc = extract_desc(file)
    desc_map: dict[str, Any] = desc[-1] if isinstance(desc[-1], dict) else dict()
    metrics = result["metrics"][0]
    return {
        "Algorithm": algorithm,
        "Inserted": metrics.get("inserted", 0),
        "Reinserted": metrics.get("reinserted", 0),
        "Miss Ratio": metrics.get("miss_ratio", 0),
        "Hit": metrics.get("hit", 0),
        "Request": metrics.get("req", 0),
        "P": float(cast(str, desc_map.get("p", nan))),
        "Trace": os.path.basename(prefix),
        "Trace Path": desc_map.get("path", "").replace("%2F", "/"),
        "Cache Size": float(cast(str, desc[0])),
        "Ignore Obj Size": desc.count("ignore_obj_size"),
        "JSON File": os.path.basename(file),
    }


def GetResult(paths: List[str], plot_name: str, index=0):
    tmp = []
    for file in paths:
        if Path(file).stat().st_size == 0:
            continue
        f = open(file, "r")
        j = json.load(f)
        f.close()
        r = ProcessResultJSON(j["results"][index], file, plot_name)
        tmp.append(r)
    return pd.DataFrame(tmp)
