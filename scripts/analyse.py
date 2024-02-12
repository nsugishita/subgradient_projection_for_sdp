# -*- coding: utf-8 -*-

"""Description of this file"""

import copy
import argparse
import logging
import os
import pickle
import typing

import pandas as pd
import numpy as np
import yaml


input_file_paths1 = [
    "data/SDPLIB/data/gpp250-1.dat-s",
    "data/SDPLIB/data/gpp250-2.dat-s",
    "data/SDPLIB/data/gpp250-3.dat-s",
    "data/SDPLIB/data/gpp250-4.dat-s",
    "data/SDPLIB/data/gpp500-1.dat-s",
    "data/SDPLIB/data/gpp500-2.dat-s",
    "data/SDPLIB/data/gpp500-3.dat-s",
    "data/SDPLIB/data/gpp500-4.dat-s",
    "data/SDPLIB/data/mcp250-1.dat-s",
    "data/SDPLIB/data/mcp250-2.dat-s",
    "data/SDPLIB/data/mcp250-3.dat-s",
    "data/SDPLIB/data/mcp250-4.dat-s",
    "data/SDPLIB/data/mcp500-1.dat-s",
    "data/SDPLIB/data/mcp500-2.dat-s",
    "data/SDPLIB/data/mcp500-3.dat-s",
    "data/SDPLIB/data/mcp500-4.dat-s",
]

input_file_paths2 = [
    "data/rudy/out/graph_1000_5_1.dat-s",
    "data/rudy/out/graph_1000_5_2.dat-s",
    "data/rudy/out/graph_1000_5_3.dat-s",
    "data/rudy/out/graph_1000_5_4.dat-s",
    "data/rudy/out/graph_2000_5_1.dat-s",
    "data/rudy/out/graph_2000_5_2.dat-s",
    "data/rudy/out/graph_2000_5_3.dat-s",
    "data/rudy/out/graph_2000_5_4.dat-s",
    "data/rudy/out/graph_3000_5_1.dat-s",
    "data/rudy/out/graph_3000_5_2.dat-s",
    "data/rudy/out/graph_3000_5_3.dat-s",
    "data/rudy/out/graph_3000_5_4.dat-s",
    "data/rudy/out/graph_4000_5_1.dat-s",
    "data/rudy/out/graph_4000_5_2.dat-s",
    "data/rudy/out/graph_4000_5_3.dat-s",
    "data/rudy/out/graph_4000_5_4.dat-s",
    "data/rudy/out/graph_5000_5_1.dat-s",
    "data/rudy/out/graph_5000_5_2.dat-s",
    "data/rudy/out/graph_5000_5_3.dat-s",
    "data/rudy/out/graph_5000_5_4.dat-s",
    "data/rudy/out/graph_1000_10_1.dat-s",
    "data/rudy/out/graph_1000_10_2.dat-s",
    "data/rudy/out/graph_1000_10_3.dat-s",
    "data/rudy/out/graph_1000_10_4.dat-s",
    "data/rudy/out/graph_2000_10_1.dat-s",
    "data/rudy/out/graph_2000_10_2.dat-s",
    "data/rudy/out/graph_2000_10_3.dat-s",
    "data/rudy/out/graph_2000_10_4.dat-s",
    "data/rudy/out/graph_3000_10_1.dat-s",
    "data/rudy/out/graph_3000_10_2.dat-s",
    "data/rudy/out/graph_3000_10_3.dat-s",
    "data/rudy/out/graph_3000_10_4.dat-s",
    "data/rudy/out/graph_4000_10_1.dat-s",
    "data/rudy/out/graph_4000_10_2.dat-s",
    "data/rudy/out/graph_4000_10_3.dat-s",
    "data/rudy/out/graph_4000_10_4.dat-s",
    "data/rudy/out/graph_5000_10_1.dat-s",
    "data/rudy/out/graph_5000_10_2.dat-s",
    "data/rudy/out/graph_5000_10_3.dat-s",
    "data/rudy/out/graph_5000_10_4.dat-s",
    "data/rudy/out/graph_1000_15_1.dat-s",
    "data/rudy/out/graph_1000_15_2.dat-s",
    "data/rudy/out/graph_1000_15_3.dat-s",
    "data/rudy/out/graph_1000_15_4.dat-s",
    "data/rudy/out/graph_2000_15_1.dat-s",
    "data/rudy/out/graph_2000_15_2.dat-s",
    "data/rudy/out/graph_2000_15_3.dat-s",
    "data/rudy/out/graph_2000_15_4.dat-s",
    "data/rudy/out/graph_3000_15_1.dat-s",
    "data/rudy/out/graph_3000_15_2.dat-s",
    "data/rudy/out/graph_3000_15_3.dat-s",
    "data/rudy/out/graph_3000_15_4.dat-s",
    "data/rudy/out/graph_4000_15_1.dat-s",
    "data/rudy/out/graph_4000_15_2.dat-s",
    "data/rudy/out/graph_4000_15_3.dat-s",
    "data/rudy/out/graph_4000_15_4.dat-s",
    "data/rudy/out/graph_5000_15_1.dat-s",
    "data/rudy/out/graph_5000_15_2.dat-s",
    "data/rudy/out/graph_5000_15_3.dat-s",
    "data/rudy/out/graph_5000_15_4.dat-s",
    "data/rudy/out/graph_1000_20_1.dat-s",
    "data/rudy/out/graph_1000_20_2.dat-s",
    "data/rudy/out/graph_1000_20_3.dat-s",
    "data/rudy/out/graph_1000_20_4.dat-s",
    "data/rudy/out/graph_2000_20_1.dat-s",
    "data/rudy/out/graph_2000_20_2.dat-s",
    "data/rudy/out/graph_2000_20_3.dat-s",
    "data/rudy/out/graph_2000_20_4.dat-s",
    "data/rudy/out/graph_3000_20_1.dat-s",
    "data/rudy/out/graph_3000_20_2.dat-s",
    "data/rudy/out/graph_3000_20_3.dat-s",
    "data/rudy/out/graph_3000_20_4.dat-s",
    "data/rudy/out/graph_4000_20_1.dat-s",
    "data/rudy/out/graph_4000_20_2.dat-s",
    "data/rudy/out/graph_4000_20_3.dat-s",
    "data/rudy/out/graph_4000_20_4.dat-s",
    "data/rudy/out/graph_5000_20_1.dat-s",
    "data/rudy/out/graph_5000_20_2.dat-s",
    "data/rudy/out/graph_5000_20_3.dat-s",
    "data/rudy/out/graph_5000_20_4.dat-s",
];

# input_file_paths2 = [
#     "data/rudy/out/weighted_graph_1000_5_1.dat-s",
#     "data/rudy/out/weighted_graph_1000_5_2.dat-s",
#     "data/rudy/out/weighted_graph_1000_5_3.dat-s",
#     "data/rudy/out/weighted_graph_1000_5_4.dat-s",
#     "data/rudy/out/weighted_graph_2000_5_1.dat-s",
#     "data/rudy/out/weighted_graph_2000_5_2.dat-s",
#     "data/rudy/out/weighted_graph_2000_5_3.dat-s",
#     "data/rudy/out/weighted_graph_2000_5_4.dat-s",
#     "data/rudy/out/weighted_graph_3000_5_1.dat-s",
#     "data/rudy/out/weighted_graph_3000_5_2.dat-s",
#     # "data/rudy/out/weighted_graph_3000_5_3.dat-s",
#     # "data/rudy/out/weighted_graph_3000_5_4.dat-s",
#     # "data/rudy/out/weighted_graph_4000_5_1.dat-s",
#     # "data/rudy/out/weighted_graph_4000_5_2.dat-s",
#     # "data/rudy/out/weighted_graph_4000_5_3.dat-s",
#     # "data/rudy/out/weighted_graph_4000_5_4.dat-s",
#     # "data/rudy/out/weighted_graph_5000_5_1.dat-s",
#     # "data/rudy/out/weighted_graph_5000_5_2.dat-s",
#     # "data/rudy/out/weighted_graph_5000_5_3.dat-s",
#     # "data/rudy/out/weighted_graph_5000_5_4.dat-s",
#     # "data/rudy/out/weighted_graph_1000_10_1.dat-s",
#     # "data/rudy/out/weighted_graph_1000_10_2.dat-s",
#     # "data/rudy/out/weighted_graph_1000_10_3.dat-s",
#     # "data/rudy/out/weighted_graph_1000_10_4.dat-s",
#     # "data/rudy/out/weighted_graph_2000_10_1.dat-s",
#     # "data/rudy/out/weighted_graph_2000_10_2.dat-s",
#     # "data/rudy/out/weighted_graph_2000_10_3.dat-s",
#     # "data/rudy/out/weighted_graph_2000_10_4.dat-s",
#     # "data/rudy/out/weighted_graph_3000_10_1.dat-s",
#     # "data/rudy/out/weighted_graph_3000_10_2.dat-s",
#     # "data/rudy/out/weighted_graph_3000_10_3.dat-s",
#     # "data/rudy/out/weighted_graph_3000_10_4.dat-s",
#     # "data/rudy/out/weighted_graph_4000_10_1.dat-s",
#     # "data/rudy/out/weighted_graph_4000_10_2.dat-s",
#     # "data/rudy/out/weighted_graph_4000_10_3.dat-s",
#     # "data/rudy/out/weighted_graph_4000_10_4.dat-s",
#     # "data/rudy/out/weighted_graph_5000_10_1.dat-s",
#     # "data/rudy/out/weighted_graph_5000_10_2.dat-s",
#     # "data/rudy/out/weighted_graph_5000_10_3.dat-s",
#     # "data/rudy/out/weighted_graph_5000_10_4.dat-s",
#     # "data/rudy/out/weighted_graph_1000_15_1.dat-s",
#     # "data/rudy/out/weighted_graph_1000_15_2.dat-s",
#     # "data/rudy/out/weighted_graph_1000_15_3.dat-s",
#     # "data/rudy/out/weighted_graph_1000_15_4.dat-s",
#     # "data/rudy/out/weighted_graph_2000_15_1.dat-s",
#     # "data/rudy/out/weighted_graph_2000_15_2.dat-s",
#     # "data/rudy/out/weighted_graph_2000_15_3.dat-s",
#     # "data/rudy/out/weighted_graph_2000_15_4.dat-s",
#     # "data/rudy/out/weighted_graph_3000_15_1.dat-s",
#     # "data/rudy/out/weighted_graph_3000_15_2.dat-s",
#     # "data/rudy/out/weighted_graph_3000_15_3.dat-s",
#     # "data/rudy/out/weighted_graph_3000_15_4.dat-s",
#     # "data/rudy/out/weighted_graph_4000_15_1.dat-s",
#     # "data/rudy/out/weighted_graph_4000_15_2.dat-s",
#     # "data/rudy/out/weighted_graph_4000_15_3.dat-s",
#     # "data/rudy/out/weighted_graph_4000_15_4.dat-s",
#     # "data/rudy/out/weighted_graph_5000_15_1.dat-s",
#     # "data/rudy/out/weighted_graph_5000_15_2.dat-s",
#     # "data/rudy/out/weighted_graph_5000_15_3.dat-s",
#     # "data/rudy/out/weighted_graph_5000_15_4.dat-s",
#     # "data/rudy/out/weighted_graph_1000_20_1.dat-s",
#     # "data/rudy/out/weighted_graph_1000_20_2.dat-s",
#     # "data/rudy/out/weighted_graph_1000_20_3.dat-s",
#     # "data/rudy/out/weighted_graph_1000_20_4.dat-s",
#     # "data/rudy/out/weighted_graph_2000_20_1.dat-s",
#     # "data/rudy/out/weighted_graph_2000_20_2.dat-s",
#     # "data/rudy/out/weighted_graph_2000_20_3.dat-s",
#     # "data/rudy/out/weighted_graph_2000_20_4.dat-s",
#     # "data/rudy/out/weighted_graph_3000_20_1.dat-s",
#     # "data/rudy/out/weighted_graph_3000_20_2.dat-s",
#     # "data/rudy/out/weighted_graph_3000_20_3.dat-s",
#     # "data/rudy/out/weighted_graph_3000_20_4.dat-s",
#     # "data/rudy/out/weighted_graph_4000_20_1.dat-s",
#     # "data/rudy/out/weighted_graph_4000_20_2.dat-s",
#     # "data/rudy/out/weighted_graph_4000_20_3.dat-s",
#     # "data/rudy/out/weighted_graph_4000_20_4.dat-s",
#     # "data/rudy/out/weighted_graph_5000_20_1.dat-s",
#     # "data/rudy/out/weighted_graph_5000_20_2.dat-s",
#     # "data/rudy/out/weighted_graph_5000_20_3.dat-s",
#     # "data/rudy/out/weighted_graph_5000_20_4.dat-s",
# ];

def main():
    """Run the main routine of this script"""
    sep = "&"
    print(f"{'problem':>8s} ", end="")
    for tol_i, tol in enumerate([1e-2, 1e-3]):
        for name in ["min", "comb", "both"]:
            print(f" {sep} ", end="")
            print(f"{name:>6s} ", end="")
    print()
    for input_file_path in input_file_paths1:
        problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
        print(f"{problem_name:8s} ", end="")
        for tol_i, tol in enumerate([1e-2, 1e-3]):
            for n_linear_cuts, eigen_comb_cut in [(1, 0), (0, 1), (1, 1)]:
                print(f" {sep} ", end="")
                output_file_path = f"outputs/v2/subgradient_projection/{problem_name}_tol_{tol}_comb_{eigen_comb_cut}_linear_{n_linear_cuts}.pkl"
                with open(output_file_path, "rb") as f:
                    res = pickle.load(f)
                print(f"{res['walltime']:6.1f} ", end="")
        print()

    print(f"{'problem':>24s} ", end="")
    for tol in [1e-2, 1e-3]:
        for solver in ["sbgrd", "mosek", "cosmo", "cosmod", "sdpnal"]:
            print(f" {sep} ", end="")
            print(f"{solver:>6s} ", end="")
    print()
    for input_file_path in input_file_paths1:
        problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
        print(f"{problem_name:>24s} ", end="")
        for tol in [1e-2, 1e-3]:
            for solver in ["subgradient_projection", "mosek", "cosmo", "cosmod", "sdpnal"]:
                print(f" {sep} ", end="")
                walltime = load_walltime(input_file_path, tol, solver)
                print(f"{walltime:6.1f} ", end="")
        print()

    print(f"{'size':>4s} ", end="")
    print(f" {sep} ", end="")
    print(f"{'densty':>6s} ", end="")
    print(f" {sep} ", end="")
    print(f"{'instance':>8s} ", end="")
    for tol in [1e-2, 1e-3]:
        for solver in ["sbgrd", "mosek"]:
            print(f" {sep} ", end="")
            print(f"{solver:>6s} ", end="")
    print()
    for input_file_path in input_file_paths2:
        problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
        _, size, density, random_seed = problem_name.split("_")
        print(f"{size:4s} ", end="")
        print(f" {sep} ", end="")
        print(f"{density:>6s} ", end="")
        print(f" {sep} ", end="")
        print(f"{random_seed:>8s} ", end="")
        for tol in [1e-2, 1e-3]:
            for solver in ["subgradient_projection", "mosek"]:
                print(f" {sep} ", end="")
                walltime = load_walltime(input_file_path, tol, solver)
                print(f"{walltime:6.1f} ", end="")
        print()


def load_walltime(input_file_path, tol, solver):
    if solver in ["subgradient_projection", "mosek"]:
        problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_path = f"outputs/v2/{solver}/{problem_name}_tol_{tol}_comb_1_linear_1.pkl"
        with open(output_file_path, "rb") as f:
            res = pickle.load(f)
        return res["walltime"]
    if solver == "cosmo":
        while True:
            problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
            if problem_name == "gpp500-3":
                return np.nan
            output_file_path = f"outputs/v2/cosmo/find_n_iterations/{problem_name}_tol_{tol}.txt"
            if os.path.exists(output_file_path):
                with open(output_file_path, "r") as f:
                    res = yaml.safe_load(f)
                return res["walltime"]
            if tol >= 1e-1:
                return np.nan
            tol *= 10
    if solver == "cosmo":
        try:
            problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
            if problem_name == "gpp500-3":
                return np.nan
            output_file_path = f"outputs/v2/cosmo/results/{problem_name}_tol_{tol}.txt"
            with open(output_file_path, "r") as f:
                res = yaml.safe_load(f)
            return res["walltime"]
        except FileNotFoundError:
            return np.nan
    if solver == "cosmod":
        try:
            problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
            output_file_path = f"outputs/v2/cosmo_jump_original/{problem_name}_tol_{tol}.txt"
            with open(output_file_path, "r") as f:
                res = yaml.safe_load(f)
            return res["walltime"]
        except FileNotFoundError:
            return np.nan
    if solver == "sdpnal":
        problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_path = f"outputs/v2/sdpnal/{problem_name}_{tol}_0.001.npz"
        res = np.load(output_file_path)
        return res["walltime"][-1]


if __name__ == "__main__":
    main()
