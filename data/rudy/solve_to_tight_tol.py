# -*- coding: utf-8 -*-

"""Description of this file"""

import argparse
from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
import pickle
from cpsdppy.sdp_solvers import cosmo, mosek, subgradient_projection
import os

dir_name = "data/rudy/out/"

os.makedirs(dir_name, exist_ok=True)

def run(sizes, densities, random_seeds):
    if isinstance(sizes, int):
        sizes = [sizes]
    if isinstance(densities, int):
        densities = [densities]
    if isinstance(random_seeds, int):
        random_seeds = [random_seeds]

    config = config_module.Config()
    config.tol = 1e-6
    config.feas_tol = 1e-6

    for size in sizes:
        for density in densities:
            for random_seed in random_seeds:
                result_file_name = f"graph_{size}_{density}_{random_seed}.pkl"
                if os.path.exists(dir_name + result_file_name):
                    continue
                input_file_name = f"graph_{size}_{density}_{random_seed}.dat-s"
                print(input_file_name)
                with open(dir_name + input_file_name, "r") as f:
                    problem_data = sdpa.read("".join(f.readlines()).strip())
                res = mosek.run(problem_data, config)
                with open(dir_name + result_file_name, "wb") as f:
                    pickle.dump(res, f)

def main():
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--density",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        nargs="+",
        required=True,
    )
    args = parser.parse_args()
    print(args)

    run(args.size, args.density, args.random_seed)



if __name__ == "__main__":
    main()
