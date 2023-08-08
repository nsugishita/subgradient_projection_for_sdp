#!/bin/sh
#
# Set up data files to run COSMO
#
# This must be run after `download.sh`. To run the script, type
# the following in `data` directory.
#
# $ ./setup_julia_data.sh

set -e

if [[ -z "${PREFIX}" ]]; then
  MY_PREFIX="/usr"
else
  MY_PREFIX="${PREFIX}"
fi

JULIA="${PREFIX}/bin/julia"

if [ ! -d "SDPLib_Importer" ]; then
    git clone https://github.com/migarstka/SDPLib_Importer
fi

pushd SDPLib_Importer
sed -i '' -e "s/20:113/32:123/" SDPLib_Importer.jl
sed -i '' -e "s/split(ln\[1\])\[4\]/split(ln\[1\])\[8\]/" SDPLib_Importer.jl

if [ ! -d "sdplib" ]; then
    # Set up sdplib data directory.
    ln -s ../SDPLIB/data/ sdplib
    pushd ../SDPLIB/data
    ln -s ../README.md README
    popd
fi

$JULIA --project=env -e "import Pkg; Pkg.add([\"FileIO\", \"JLD2\"])"
$JULIA --project=env -e "include(\"SDPLib_Importer.jl\")"
