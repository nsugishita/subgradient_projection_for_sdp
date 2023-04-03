#!/bin/sh

# Build extensions. Call this script from the top directory of the project.
# . ./scripts/build.sh

. ./scripts/activate.sh

pushd extensions/indexremove/

rm -rf build
mkdir build
cd build
CXX=~/local/bin/g++ ~/local/bin/cmake .. -DCMAKE_INSTALL_PREFIX="$PREFIX"
make
popd

python -c "import indexremove"
echo "built extensions successfully"
