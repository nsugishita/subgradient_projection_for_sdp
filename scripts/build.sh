#!/bin/sh

# Build extensions. Call this script from the top directory of the project.
# . ./scripts/build.sh

set -e

. ./scripts/activate.sh

pushd extensions/indexremove/
mkdir -p build
cd build
CXX=~/local/default/bin/g++ ~/local/default/bin/cmake ..
make
popd
python -c "import indexremove"
echo "built 'indexremove' successfully"

pushd extensions/uniquelist/
mkdir -p build
cd build
CXX=~/local/default/bin/g++ ~/local/default/bin/cmake ..
make
popd
python -c "import uniquelist"
echo "built 'uniquelist' successfully"
