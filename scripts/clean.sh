#!/bin/sh

# Build caches to build extensions. Call this script from the top directory
# of the project.
# . ./scripts/clean.sh

set -e

. ./scripts/activate.sh

pushd extensions/indexremove/
rm -rf build
popd

pushd extensions/uniquelist/
rm -rf build
popd
