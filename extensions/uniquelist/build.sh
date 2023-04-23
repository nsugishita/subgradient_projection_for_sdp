#!/bin/sh

set -e


mkdir -p build
pushd build
/Applications/CMake.app/Contents/bin/cmake ..
make
popd

python3 test.py
echo "ok"
