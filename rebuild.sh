#! /bin/sh/

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./inst
make install