import sys

import numpy as np
import pandas as pd

KEYS = ["site", "target_month", "scale", "lead"]

def main(paths, method="droughtcast"):
    runs = {p: pd.read_csv(p) for p in paths}
    common = set.intersection(*[set(map(tuple, r[KEYS].astype(str).values)) for r in runs.values()])
    print(f"{len(common)} common (site, month, scale, lead) targets")
    for name, df in runs.items():
        keep = [tuple(x) in common for x in df[KEYS].astype(str).values]
        m = df[keep]
        err = (m[f"prediction_{method}"] - m["observed"]) ** 2
        print(name, np.sqrt(err.groupby([m["scale"], m["lead"]]).mean()).round(3).to_string())

if __name__ == "__main__":
    main(sys.argv[1:])