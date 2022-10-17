# BlosSOM :blossom:

## Compilation
Install some alternative of `libglfw3-dev` and `libglm-dev`.

```sh
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./inst
make install
```

#### Running
```sh
./inst/bin/blossom
```
