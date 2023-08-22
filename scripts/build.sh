#!/bin/sh

# Build extensions. Call this script from the top directory of the project.
# . ./scripts/build.sh

set -e

. ./scripts/activate.sh

if [[ -z "${PREFIX}" ]]; then
  MY_PREFIX="/usr"
else
  MY_PREFIX="${PREFIX}"
fi

CXX="$MY_PREFIX/bin/g++"
CMAKE="$MY_PREFIX/bin/cmake"

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
