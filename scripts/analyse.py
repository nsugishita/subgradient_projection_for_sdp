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
    "data/rudy/out/gpp_1000_5_1.dat-s",
    "data/rudy/out/gpp_1000_5_2.dat-s",
    "data/rudy/out/gpp_1000_5_3.dat-s",
    "data/rudy/out/gpp_1000_5_4.dat-s",
    "data/rudy/out/gpp_2000_5_1.dat-s",
    "data/rudy/out/gpp_2000_5_2.dat-s",
    "data/rudy/out/gpp_2000_5_3.dat-s",
    "data/rudy/out/gpp_2000_5_4.dat-s",
    "data/rudy/out/gpp_3000_5_1.dat-s",
    "data/rudy/out/gpp_3000_5_2.dat-s",
    "data/rudy/out/gpp_3000_5_3.dat-s",
    "data/rudy/out/gpp_3000_5_4.dat-s",
    "data/rudy/out/gpp_4000_5_1.dat-s",
    "data/rudy/out/gpp_4000_5_2.dat-s",
    "data/rudy/out/gpp_4000_5_3.dat-s",
    "data/rudy/out/gpp_4000_5_4.dat-s",
    "data/rudy/out/gpp_5000_5_1.dat-s",
    "data/rudy/out/gpp_5000_5_2.dat-s",
    "data/rudy/out/gpp_5000_5_3.dat-s",
    "data/rudy/out/gpp_5000_5_4.dat-s",
    "data/rudy/out/gpp_1000_10_1.dat-s",
    "data/rudy/out/gpp_1000_10_2.dat-s",
    "data/rudy/out/gpp_1000_10_3.dat-s",
    "data/rudy/out/gpp_1000_10_4.dat-s",
    "data/rudy/out/gpp_2000_10_1.dat-s",
    "data/rudy/out/gpp_2000_10_2.dat-s",
    "data/rudy/out/gpp_2000_10_3.dat-s",
    "data/rudy/out/gpp_2000_10_4.dat-s",
    "data/rudy/out/gpp_3000_10_1.dat-s",
    "data/rudy/out/gpp_3000_10_2.dat-s",
    "data/rudy/out/gpp_3000_10_3.dat-s",
    "data/rudy/out/gpp_3000_10_4.dat-s",
    "data/rudy/out/gpp_4000_10_1.dat-s",
    "data/rudy/out/gpp_4000_10_2.dat-s",
    "data/rudy/out/gpp_4000_10_3.dat-s",
    "data/rudy/out/gpp_4000_10_4.dat-s",
    "data/rudy/out/gpp_5000_10_1.dat-s",
    "data/rudy/out/gpp_5000_10_2.dat-s",
    "data/rudy/out/gpp_5000_10_3.dat-s",
    "data/rudy/out/gpp_5000_10_4.dat-s",
];

def emphasize(text, mode=True):
    if mode:
        return "\\textbf{" + text + "}"
    else:
        return "        " + text + " "

def main():
    """Run the main routine of this script"""
    for item in ["walltime", "n_iterations"]:
        print(f"item: {item}")
        sep = "&"
        print(f"{'problem':>8s} ", end="")
        for tol_i, tol in enumerate([1e-2, 1e-3]):
            for name in ["min", "comb", "both"]:
                print(f" {sep} ", end="")
                print(emphasize(f"{name:>6s}", 0) + " ", end="")
        print("\\\\")
        for input_file_path in input_file_paths1:
            problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
            print(f"{problem_name:8s} ", end="")
            for tol_i, tol in enumerate([1e-2, 1e-3]):
                values = []
                for n_linear_cuts, eigen_comb_cut in [(1, 0), (0, 1), (1, 1)]:
                    output_file_path = f"outputs/v2/subgradient_projection/{problem_name}_tol_{tol}_comb_{eigen_comb_cut}_linear_{n_linear_cuts}.pkl"
                    with open(output_file_path, "rb") as f:
                        res = pickle.load(f)
                    values.append(res[item])
                min_ = min([np.round(x, 1) for x in values])
                if item == "walltime":
                    values = [emphasize(f"{x:6.1f}", np.round(x, 1) == min_) for x in values]
                elif item == "n_iterations":
                    values = [emphasize(f"{x:6d}", np.round(x, 1) == min_) for x in values]
                for x in values:
                    print(f" {sep} ", end="")
                    print(f"{x} ", end="")
            print("\\\\")

    for item in ["walltime", "n_iterations"]:
        print(f"{'problem':>24s} ", end="")
        for tol in [1e-2, 1e-3]:
            for solver in ["sbgrd", "mosek", "cosmo", "cosmon", "sdpnal"]:
                if solver == "cosmon":
                    continue
                print(f" {sep} ", end="")
                print(emphasize(f"{solver:>7s}", 0) + " ", end="")
        print("\\\\")
        for input_file_path in input_file_paths1:
            problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
            print(f"{problem_name:>24s} ", end="")
            for tol in [1e-2, 1e-3]:
                values = []
                for solver in ["subgradient_projection", "mosek", "cosmo", "cosmon", "sdpnal"]:
                    if solver == "cosmon":
                        continue
                    v = load_walltime(input_file_path, tol, solver, item)
                    values.append(v)
                min_ = min([np.round(x, 1) for x in values])
                values = [emphasize(f"{x:7.1f}", np.round(x, 1) == min_) for x in values]
                for x in values:
                    print(f" {sep} ", end="")
                    print(x + " ", end="")
            print("\\\\")

    for problem_name in ["gpp", "graph"]:
        print(f"problem: {problem_name}")
        # print(f"{'':>6s} ", end="")
        # print(f" {sep} ", end="")
        print(f"{'':>4s} ", end="")
        print(f" {sep} ", end="")
        print(f"{'':>8s} ", end="")
        for density in [5, 10]:
            for tol_i, tol in enumerate([1e-2, 1e-3]):
                for solver_i, solver in enumerate(["sbgrd", "mosek"]):
                    print(f" {sep} ", end="")
                    if tol_i == 0 and solver_i == 0:
                        print(emphasize(f"d: {density:3d}", 0) + " ", end="")
                    else:
                        print(emphasize(f"{'':>6s}", 0) + " ", end="")
        print("\\\\")
        # print(f"{'densty':>6s} ", end="")
        # print(f" {sep} ", end="")
        print(f"{'size':>4s} ", end="")
        print(f" {sep} ", end="")
        print(f"{'instance':>8s} ", end="")
        for density in [5, 10]:
            for tol in [1e-2, 1e-3]:
                for solver in ["sbgrd", "mosek"]:
                    print(f" {sep} ", end="")
                    print(emphasize(f"{solver:>6s}", 0) + " ", end="")
        print("\\\\")
        for size in [1000, 2000, 3000, 4000, 5000]:
            for random_seed in [1, 2, 3, 4]:
                # print(f"{problem_name:>7s} ", end="")
                # print(f"{density:6d} ", end="")
                # print(f" {sep} ", end="")
                print(f"{size:4d} ", end="")
                print(f" {sep} ", end="")
                print(f"{random_seed:8d} ", end="")
                for density in [5, 10]:
                    input_file_path = f"data/rudy/out/{problem_name}_{size}_{density}_{random_seed}.dat-s"
                    for tol in [1e-2, 1e-3]:
                        values = []
                        for solver in ["subgradient_projection", "mosek"]:
                            values.append(load_walltime(input_file_path, tol, solver))
                        min_ = min([np.round(x, 1) for x in values])
                        values = [emphasize(f"{x:6.1f}", np.round(x, 1) == min_) for x in values]
                        for x in values:
                            print(f" {sep} ", end="")
                            print(f"{x} ", end="")
                print("\\\\")


def load_walltime(input_file_path, tol, solver, item="walltime"):
    if (item == "walltime") and (tol == 1e-3):
        n1 = load_walltime(input_file_path, 1e-2, solver, item="n_iterations")
        n2 = load_walltime(input_file_path, 1e-3, solver, item="n_iterations")
        if n1 == n2:
            return load_walltime(input_file_path, 1e-2, solver, item)
    if solver in ["subgradient_projection", "mosek"]:
        problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_path = f"outputs/v2/{solver}/{problem_name}_tol_{tol}_comb_1_linear_0.pkl"
        if not os.path.isfile(output_file_path):
            output_file_path = f"outputs/v2/{solver}/{problem_name}_tol_{tol}_comb_1_linear_1.pkl"
        with open(output_file_path, "rb") as f:
            res = pickle.load(f)
        return res[item]
    if solver == "cosmon":
        while True:
            problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
            output_file_path = f"outputs/v2/cosmo/find_n_iterations2/{problem_name}_tol_{tol}.txt"
            if os.path.exists(output_file_path):
                with open(output_file_path, "r") as f:
                    res = yaml.safe_load(f)
                return res[item]
            if tol >= 1e-1:
                return np.nan
            tol *= 10
    if solver == "cosmo":
        try:
            problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
            output_file_path = f"outputs/v2/cosmo_jump_original/{problem_name}_tol_{tol}.txt"
            with open(output_file_path, "r") as f:
                res = yaml.safe_load(f)
            return res[item]
        except FileNotFoundError:
            return np.nan
    if solver == "sdpnal":
        problem_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_path = f"outputs/v2/sdpnal/{problem_name}_{tol}_0.001.npz"
        res = np.load(output_file_path)
        return res[item][-1]


if __name__ == "__main__":
    main()
