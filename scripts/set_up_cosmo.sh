#!/bin/sh
#
# Set up COSMO
#
# To run the script, type the following in the top directory.
#
# $ ./scripts/setup_cosmo.sh

set -e

if [[ -z "${PREFIX}" ]]; then
  MY_PREFIX="/usr"
else
  MY_PREFIX="${PREFIX}"
fi

JULIA="${PREFIX}/bin/julia"

$JULIA --project=juliaenv -e "import Pkg; Pkg.add([\"FileIO\", \"JLD2\", \"COSMO\", \"JuMP\", \"JSON\"])"
