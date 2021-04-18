# Interactive EmbedSOM research project

# Building

## Windows
```sh
git submodule init
git submodule update

mkdir build
cd build
# If SDL2 dependency is installed in ~source/repos/vcpkg directory
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX=./inst -DCMAKE_TOOLCHAIN_FILE=~/source/repos/vcpkg/scripts/buildsystems/vcpkg.cmake

cmake --build . --config Release
cmake --install . --config Release
```

## Unix-like systems
```sh
git submodule init
git submodule update

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./inst
make install
```

# Controls
Move: Arrows/Mouse drag

Zoom in: Esc/Mouse wheel

Zoom out: Space/Mouse wheel
