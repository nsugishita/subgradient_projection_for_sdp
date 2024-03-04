#!/bin/sh
#
# Set up COSMO
#
# To run the script, type the following in the top directory.
#
# $ ./scripts/setup_cosmo.sh

set -e

JULIA=bin/julia

$JULIA --project=juliaenv -e "import Pkg; Pkg.add([\"COSMO\", \"JuMP\"])"
