#!/bin/sh

# Initialise env vars.
# Call this script from the top directory of this project.
# . scripts/init.sh

export PYTHONPATH="$PYTHONPATH":"$(pwd)/extensions/indexremove/build"
export PYTHONPATH="$PYTHONPATH":"$(pwd)/extensions/uniquelist/build"
export PREFIX="$HOME/local/default"
export OMP_NUM_THREADS=1

module load gurobi >/dev/null 2>&1 || true
module load julia >/dev/null  2>&1|| true

. ./env/bin/activate
