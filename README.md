# BlosSOM :blossom:

## Compilation
Install some alternative of `libglfw3-dev` and `libglm-dev`.

```sh
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./inst
# to compile cuda, set proper gcc/g++ compiler compatible with your version of cuda/nvcc
# cmake .. -DCMAKE_INSTALL_PREFIX=./inst -DBUILD_CUDA=1 -DCMAKE_C_COMPILER=/usr/bin/gcc-10 -DCMAKE_CXX_COMPILER=/usr/bin/g++-10
make install
```



#### Running
```sh
./inst/bin/blossom
# ./inst/bin/blossom_cuda
```

