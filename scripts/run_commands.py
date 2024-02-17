# -*- coding: utf-8 -*-

"""Run commands in a given file in parallel"""

import random
import multiprocessing as mp
import os
import argparse
import subprocess
import sys


def worker(path, worker_index, lock, shuffle):
    iteration_index = -1

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    while True:
        iteration_index += 1
        with lock:
            with open(path, "r") as f:
                read = f.read().strip()  # read everything in the file
                if not read:
                    return
                lines = read.split("\n")
            if shuffle:
                random.shuffle(lines)
            # Find the first non-empty line with any first character except '#'.
            found = False
            while True:
                if not lines:
                    break
                line = lines[0].strip()
                if line and line[0] != "#":
                    found = True
                    break
                lines = lines[1:]
            if not found:
                break
            with open(path, "w") as f:
                for x in lines[1:]:
                    f.write(x + "\n")

        print(f"worker: {worker_index}   command: {line}")
        subprocess.run(line, shell=True, check=True, capture_output=True, env=env)


def main():
    """Run the main routine of this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to a file with commands")
    parser.add_argument("--n", type=int, default=4, help="number of processes")
    parser.add_argument("--shuffle", action="store_true", help="shuffle commands")
    args = parser.parse_args()

    lock = mp.Lock()
    processes = []

    for worker_index in range(args.n):
        process = mp.Process(target=worker, args=(args.path, worker_index, lock, args.shuffle))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
