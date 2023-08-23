#!/bin/sh

# Build extensions. Call this script from the top directory of the project.
# . ./scripts/build.sh

set -e

. ./scripts/activate.sh

CXX=$(pwd)/bin/g++
CMAKE=$(pwd)/bin/cmake

pushd extensions/indexremove/
mkdir -p build
cd build
CXX=$CXX $CMAKE ..
make
popd
python -c "import indexremove"
echo "built 'indexremove' successfully"

pushd extensions/uniquelist/
mkdir -p build
cd build
CXX=$CXX $CMAKE ..
make
popd
python -c "import uniquelist"
echo "built 'uniquelist' successfully"
