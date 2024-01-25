# -*- coding: utf-8 -*-

"""Solve SDPA

This scripts run the subgradient projection solver or MOSEK
to solve a test instance in SDPLIB.

The following two functions are defined in this script.

- run(config, dir):
    Run a solver in the same process.

Additionally, one can run a solver from the command line.

```
$ python examples/solve_sdpa.py --problem-name theta1 --solver mosek
```

"""

import argparse
import numpy as np
import logging
import os
import pickle
import typing

from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
from cpsdppy.sdp_solvers import common
from cpsdppy.sdp_solvers import cosmo, mosek, subgradient_projection

logger = logging.getLogger(__name__)

missing = {}

def main() -> None:
    """Run the entry point of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--solution",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    config = config_module.Config()
    config.problem_name = args.problem

    x = np.load(args.solution)

    problem_data = sdpa.read(config)
    eval_x = common.evaluate_solution(x, problem_data)
    print(f"{eval_x.f} {eval_x.f_gap} {eval_x.g.item()}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
