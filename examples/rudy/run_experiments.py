# -*- coding: utf-8 -*-

"""Description of this file"""

import argparse
import solve_to_tight_tol
import solve

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

    solve_to_tight_tol.run(args.size, args.density, args.random_seed)
    solve.run(args.size, args.density, args.random_seed, args.tol, args.mosek, args.subgrad, args.cosmo)



if __name__ == "__main__":
    main()
