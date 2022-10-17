#! /bin/sh/

rm -rf build
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./inst -DBUILD_CUDA=1
make install
