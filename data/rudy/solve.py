# -*- coding: utf-8 -*-

"""Description of this file"""

import argparse
from cpsdppy import config as config_module
from cpsdppy import logging_helper, sdpa
import pickle
from cpsdppy.sdp_solvers import cosmo, mosek, subgradient_projection
import os

input_dir_name = "data/rudy/out/"
output_dir_name = "data/rudy/res/"

os.makedirs(input_dir_name, exist_ok=True)
os.makedirs(output_dir_name, exist_ok=True)

def run(sizes, densities, random_seeds, tols, mosek, subgradient_projection, cosmo):
    if isinstance(sizes, int):
        sizes = [sizes]
    if isinstance(densities, int):
        densities = [densities]
    if isinstance(random_seeds, int):
        random_seeds = [random_seeds]
    if isinstance(tols, float):
        tols = [tols]
    solvers = []
    if mosek:
        solvers.append("mosek")
    if subgradient_projection:
        solvers.append("subgradient_projection")
    if cosmo:
        solvers.append("cosmo")

    logging_helper.setup(dir="")

    for tol in tols:
        for size in sizes:
            for density in densities:
                for random_seed in random_seeds:
                    for solver in solvers:
                        impl(size, density, random_seed, tol, solver)


def impl(size, density, random_seed, tol, solver):
    config = config_module.Config()
    config.n_linear_cuts = 0
    config.eigen_comb_cut = 1
    config.tol = tol
    config.solver = solver

    config_str = config._asstr(only_modified=True, shorten=True)
    log_path = output_dir_name + f"size_{size}_density_{density}_random_seed_{random_seed}_" + config_str + ".txt"
    result_path = output_dir_name + f"size_{size}_density_{density}_random_seed_{random_seed}_" + config_str + ".pkl"

    if os.path.exists(result_path):
        return

    print(f"graph_{size}_{density}_{random_seed}.dat-s")
    input_problem_file_name = input_dir_name + f"graph_{size}_{density}_{random_seed}.dat-s"
    input_solution_file_name = input_dir_name + f"graph_{size}_{density}_{random_seed}.pkl"

    with open(input_problem_file_name, "r") as f:
        problem_data = sdpa.read("".join(f.readlines()).strip())

    with open(input_solution_file_name, "rb") as f:
        tight_tol_res = pickle.load(f)

    problem_data["target_objective"] = 0.5 * (tight_tol_res["primal_objective"] + tight_tol_res["dual_objective"])

    with logging_helper.save_log(log_path):
        if config.solver == "mosek":
            res = mosek.run(problem_data, config)
        elif config.solver == "subgradient_projection":
            res = subgradient_projection.run(problem_data, config)
        elif config.solver == "cosmo":

            config.problem_name = input_dir_name + f"graph_{size}_{density}_{random_seed}.jld2"
            res = cosmo.run(problem_data, config)

    with open(result_path, "wb") as f:
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
    parser.add_argument(
        "--tol",
        type=float,
        nargs="+",
        required=True
    )
    parser.add_argument(
        "--mosek",
        action="store_true",
    )
    parser.add_argument(
        "--subgrad",
        action="store_true",
    )
    parser.add_argument(
        "--cosmo",
        action="store_true",
    )
    args = parser.parse_args()

    if (not args.mosek) and (not args.subgrad) and (not args.cosmo):
        print("--mosek, --subgrad or --cosmo must be given")
        return

    run(args.size, args.density, args.random_seed, args.tol, args.mosek, args.subgrad, args.cosmo)



if __name__ == "__main__":
    main()
