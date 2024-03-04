# -*- coding: utf-8 -*-

"""Solve SDPA"""

import argparse
import subprocess

def main():
    """Run the entry point of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["subgradient_projection", "mosek", "cosmo"],
        required=True,
    )
    parser.add_argument(
        "--tol",
        type=float,
        choices=[1e-2, 1e-3],
        default=1e-3,
        help=(
            "suboptimality tolerance. Only used by subgradient_projection and "
            "mosek. The other solvers solve the specified problems with "
            "tolerance 1% and 0.1%."
        ),
    )
    parser.add_argument(
        "--comb",
        type=int,
        choices=[0, 1],
        default=1,
        help="if this is 1, subgradient_projection uses the comb cuts",
    )
    parser.add_argument(
        "--linear",
        type=int,
        choices=[0, 1],
        default=0,
        help="if this is 1, subgradient_projection uses the linear cuts",
    )
    args = parser.parse_args()

    if args.solver in ["subgradient_projection", "mosek"]:
        from internal.run_subgradient_projection_or_mosek import _impl
        _impl(args)

    elif args.solver == "cosmo":
        command = (
            ". ./scripts/activate.sh && julia --project=juliaenv "
            f"scripts/internal/run_cosmo.jl {args.problem}"
        )
        subprocess.run(command, shell=True, check=True)
    else:
        raise ValueError


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
