# -*- coding: utf-8 -*-

"""Run commands in a given file in parallel"""

import multiprocessing as mp
import os
import subprocess
import sys

n_workers = 6


def worker(path, worker_index, lock):
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
            line = lines[0]
            with open(path, "w") as f:
                for x in lines[1:]:
                    f.write(x + "\n")

        print(f"worker: {worker_index}   command: {line}")
        subprocess.run(line, shell=True, check=True, env=env)


def main():
    """Run the main routine of this script"""
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} FILE")
        return

    path = sys.argv[1]

    lock = mp.Lock()
    processes = []

    for worker_index in range(n_workers):
        process = mp.Process(target=worker, args=(path, worker_index, lock))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
