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

from cpsdppy import config as config_module
from cpsdppy import sdpa
from cpsdppy.sdp_solvers import mosek


def run(data_file_path):
    config = config_module.Config()
    config.tol = 1e-6
    config.feas_tol = 1e-6

    output_file_path = os.path.splitext(data_file_path)[0] + ".txt"

    if os.path.exists(output_file_path):
        print(f"already exists: {output_file_path}")
        return

    print(f"solving: {data_file_path}")
    with open(data_file_path, "r") as f:
        problem_data = sdpa.read("".join(f.readlines()).strip())
    res = mosek.run(problem_data, config)
    optimal_objective = 0.5 * (res["primal_objective"] + res["dual_objective"])
    print(f"optimal objective: {optimal_objective}")
    with open(output_file_path, "w") as f:
        f.write(str(optimal_objective))


def main():
    """Run the main routine of this script"""
    for data_file_path in sys.argv[1:]:
        run(data_file_path)


if __name__ == "__main__":
    main()
