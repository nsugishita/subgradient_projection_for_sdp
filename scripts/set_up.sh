#!/bin/sh

set -e

. ./scripts/activate.sh

echo "installing python packages"

pip install -e . >/dev/null 2>&1

echo "installing julia packages"

. ./scripts/internal/set_up_cosmo.sh >/dev/null 2>&1

echo "downloading SDPLIB instances"

pushd data >/dev/null
./download_sdplib.sh >/dev/null
popd >/dev/null
python scripts/internal/populate_optimal_objective_values_of_sdplib.py

echo "building C++ extensions"

./scripts/internal/build.sh >/dev/null

echo "successfully set up"
