#!/usr/bin/env sh
set -evx
pwd
ls
mkdir build || true
cd build
cmake -DBUILD_DOC=OFF \
      -DBUILD_DEMO=OFF \
      -DBUILD_TEST=OFF \
      -DBUILD_BENCH=ON \
      -DCMAKE_CXX_FLAGS=$CXX_FLAGS \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      ..
make
cd Test
CTEST_OUTPUT_ON_FAILURE=1 ctest --verbose
