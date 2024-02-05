#!/bin/sh

mkdir out

for density in 10 20 50; do
    for size in 1000 2000 3000 4000 5000 ; do
        for seed in 1 2 3 4; do
            python interface.py $size $density $seed > out/graph_${size}_${density}_${seed}.dat-s
        done
    done
done
