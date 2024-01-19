#!/bin/sh
#
# Download SDPLIB data
#
# To run the script, type the following in `data` directory.
#
# $ ./download_sdplib.sh

set -e

if [ ! -d "SDPLIB" ]; then
    git clone https://github.com/vsdp/SDPLIB
    sed -i.bu "s/eqaulG11/equalG11/" SDPLIB/README.md
fi
