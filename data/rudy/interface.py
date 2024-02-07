# -*- coding: utf-8 -*-

"""Run rudy and generate a max-cut problem


"""

import sys
import os
import subprocess


def run(size, density, random_seed, weight_lb, weight_ub):
    """Run the main routine of this script"""
    if weight_lb is not None:
        program = os.path.join(os.path.dirname(__file__), "rudy")
        command = [
            program,
            "-rnd_graph",
            str(size),
            str(density),
            str(random_seed),
            "-random",
            str(weight_lb),
            str(weight_ub),
            str(int(random_seed) + 1000),
        ]
        res = subprocess.run(
            command, capture_output=True, check=True, encoding="utf8"
        )
        return convert(res.stdout)
    else:
        program = os.path.join(os.path.dirname(__file__), "rudy")
        command = [
            program,
            "-rnd_graph",
            str(size),
            str(density),
            str(random_seed),
        ]
        res = subprocess.run(
            command, capture_output=True, check=True, encoding="utf8"
        )
        return convert(res.stdout)


def convert(text):
    lines = text.strip().split("\n")
    size, n_edges = lines[0].strip().split()
    size = int(size)
    n_edges = int(n_edges)

    coef = {i: {i: 0} for i in range(1, size + 1)}
    for line in lines[1:]:
        i, j, weight = line.strip().split()
        i, j, weight = int(i), int(j), float(weight) / 4
        coef[i][i] += weight
        coef[j][j] += weight
        coef[i][j] = -weight

    out = ""
    out += str(size) + "\n"
    out += "1\n"
    out += str(size) + "\n"
    out += "{"
    for i in range(size):
        if i > 0:
            out += ","
        out += "+1.0"
    out += "}\n"

    for i in coef:
        for j in coef[i]:
            out += f"0 1 {i} {j} {coef[i][j]}\n"

    for i in range(1, size + 1):
        out += f"{i} 1 {i} {i} 1\n"

    return out


def main():
    if len(sys.argv) != 4 and len(sys.argv) != 6:
        print(
            f"usage: python {sys.argv[0]} size density random_seed [weight_lb weight_ub]"
        )
        return

    argv = sys.argv[1:]

    if len(argv) == 3:
        argv.append(None)
        argv.append(None)
    print(run(argv[0], argv[1], argv[2], argv[3], argv[4]).strip())


if __name__ == "__main__":
    main()
