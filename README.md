# Interactive EmbedSOM Research Project

## Compilation

The project requires [`cmake`](https://cmake.org/) build system and a few other dependencies listed below.

### Windows (Visual Studio 2019)

The project requires SDL2 as an external dependency.
1) install [vcpkg](https://github.com/microsoft/vcpkg) tool and remember your vcpkg directory
2) install SDL: `vcpkg install SDL2:x64-windows`

```sh
git submodule init
git submodule update

mkdir build
cd build
# do not forget to fix path to vcpkg in the following command
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX=./inst -DCMAKE_TOOLCHAIN_FILE=your-vcpkg-clone-directory/scripts/buildsystems/vcpkg.cmake

cmake --build . --config Release
cmake --install . --config Release
```

### Linux (and possibly other unix-like systems)

```sh
git submodule init
git submodule update

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./inst
make install
```

## Controls

Move: Arrows/Mouse drag

Zoom in: Esc/Mouse wheel

Zoom out: Space/Mouse wheel
