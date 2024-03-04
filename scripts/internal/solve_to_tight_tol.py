# -*- coding: utf-8 -*-

"""Solve an SDP problem using Mosek and save the optimal objective value

This accepts the paths to data and run Mosek to get the optimal objective value.
The result will be saved in the directory where the data is saved.

For example,

```
python scripts/solve_to_tight_tol.py mydata/foo/bar.dat-s mydata/baz.dat-s
```

will solve two SDP problems and create `mydata/foo/bar.txt` and
`mydata/baz.txt`. If the files already exist, this does not do anything.
"""

import os
import sys
import random

from cpsdppy import config as config_module
from cpsdppy import sdpa
from cpsdppy.sdp_solvers import mosek


def run(data_dir):
    updated = False

    if os.path.isfile(data_dir):
        files = [data_dir]
        isfile = True
    else:
        files = list(os.listdir(data_dir))
        isfile = False

    random.shuffle(files)

    for file_name in files:
        if not file_name.endswith(".dat-s"):
            continue
        if isfile:
            data_file_path = file_name
        else:
            data_file_path = os.path.join(data_dir, file_name)
        output_file_path = os.path.splitext(data_file_path)[0] + ".txt"
        if os.path.exists(output_file_path):
            continue
        print(f"solving: {data_file_path}")
        config = config_module.Config()
        config.tol = 1e-6
        config.feas_tol = 1e-6
        config.log_to_stdout = 1
        config.time_limit = 5 * 60 * 60
        with open(data_file_path, "r") as f:
            problem_data = sdpa.read("".join(f.readlines()).strip())
        res = mosek.run(problem_data, config)
        optimal_objective = 0.5 * (res["primal_objective"] + res["dual_objective"])
        print(f"optimal objective: {optimal_objective}")
        with open(output_file_path, "w") as f:
            f.write(str(optimal_objective))
        updated = True
    return updated


def main():
    """Run the main routine of this script"""
    while True:
        updated = False
        for data_dir in sys.argv[1:]:
            print(f"scanning {data_dir}")
            updated = run(data_dir) or updated
        if not updated:
            print("new files not found")
            break


if __name__ == "__main__":
    main()
