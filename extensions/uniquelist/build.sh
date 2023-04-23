#!/bin/sh

set -e


mkdir -p build
pushd build
/Applications/CMake.app/Contents/bin/cmake ..
make
popd

PYTHONPATH="$PYTHONPATH":"$(pwd)/build" python3 test.py
echo "ok"
