# -*- coding: utf-8 -*-

"""Populate files containing the optimal objectives in SDPLIB data

To run this script run the following command from the top of the project:

```
python scripts/populate_optimal_objective_values_of_sdplib.py
```
"""

import io
import os

import pandas as pd


def main():
    """Run the main routine of this script"""
    sdplib_path = "data/SDPLIB/"

    lines = []
    with open(os.path.join(sdplib_path, "README.md"), "r") as f:
        mode = 0
        for i, l in enumerate(f):
            if mode == 0:
                if l.startswith("|"):
                    mode += 1
            if mode == 1:
                if l.startswith("|"):
                    lines.append(l)
                else:
                    mode += 1
            if mode >= 2:
                break
    lines = lines[:1] + lines[2:]
    lines = [x[2:-3].replace("|", ",") for x in lines]
    lines = "\n".join(lines)
    df = pd.read_csv(io.StringIO(lines), sep=" +, +", engine="python")
    for index, row in df.iterrows():
        problem_name = row["Problem"]
        try:
            optimal_objective = float(row["Optimal Objective Value"])
        except:
            optimal_objective = None
        if optimal_objective is None:
            continue
        output_file_path = os.path.join(
            sdplib_path, "data", f"{problem_name}.txt"
        )
        with open(output_file_path, "w") as f:
            f.write(str(optimal_objective))


if __name__ == "__main__":
    main()
