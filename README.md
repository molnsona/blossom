# BlosSOM :blossom:

BlosSOM is a graphical environment for running semi-supervised dimensionality
reduction with [EmbedSOM](https://github.com/exaexa/EmbedSOM). You can use it
to explore multidimensional datasets, and produce great-looking 2-dimensional
visualizations.

**Warning: BlosSOM is still under development**, things will magically improve without notice.

- :question: [Overview](#overview)
- :wrench: [Compiling and running](#compiling-and-running-blossom)
- :arrow_right: [How-To](HOWTO.md) :bulb:
- :blue_book: [Documentation](#documentation)

![screenshot](media/screenshot.png)

BlosSOM was developed at the [MFF UK Prague](https://www.mff.cuni.cz/en), in cooperation with [IOCB Prague](https://www.uochb.cz/).

<img src="media/mff-black-full.svg" alt="MFF logo" height="64px"> 
<img src="media/uochb.svg" alt="IOCB logo" height="64px"> 

## Overview

BlosSOM creates a landmark-based model of the dataset, and dynamically projects
all dataset point to your screen (using EmbedSOM). Several other algorithms and
tools are provided to manage the landmarks; a quick overview follows:

- High-dimensional landmark positioning:
    - Self-organizing maps
    - k-Means
- 2D landmark positioning
    - k-NN graph generation (only adds edges, not vertices)
    - force-based graph layouting
    - dynamic t-SNE
- Dimensionality reduction
    - EmbedSOM
    - CUDA EmbedSOM (with roughly 500x speedup, enabling smooth display of a few millions of points)
- Manual landmark position optimization
- Visualization settings (colors, transparencies, cluster coloring, ...)
- Dataset transformations and dimension scaling
- Import from matrix-like data files
  - FCS3.0 (Flow Cytometry Standard files)
  - TSV (Tab-separated CSV)
- Export of the data for plotting

## Compiling and running BlosSOM

For CUDA EmbedSOM to work, you need the [NVIDIA CUDA
toolkit](https://developer.nvidia.com/cuda-zone). You also need
[cmake](https://cmake.org/) build system and [SDL2](https://www.libsdl.org/).

### Windows (Visual Studio 2019)

#### Dependencies
The project requires SDL2 as an external dependency:

1. install [vcpkg](https://github.com/microsoft/vcpkg) tool and remember your vcpkg directory
2. install SDL: `vcpkg install SDL2:x64-windows`

#### Compilation
```sh
git submodule init
git submodule update

mkdir build
cd build

# You need to fix the path to vcpkg in the following command:
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX=./inst -DCMAKE_TOOLCHAIN_FILE=your-vcpkg-clone-directory/scripts/buildsystems/vcpkg.cmake

cmake --build . --config Release
cmake --install . --config Release
```

#### Running
Open Visual Studio solution `BlosSOM.sln`, set `blossom` as startup project,
set configuration to Release and run the project.

### Linux (and possibly other unix-like systems)

#### Dependencies
The project requires SDL2 as an external dependency. Install `libsdl2-dev` (on
Debian-based systems) or `SDL2-devel` (on Red Hat-based systems), or similar
(depending on the Linux distribution). You should be able to install `cmake`
package the same way.

#### Compilation
```sh
git submodule init
git submodule update

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./inst    # or any other directory
make install                              # use -j option to speed up the build
```

#### Running
```sh
./inst/bin/blossom        # this runs the slower (but compatible) CPU version
./inst/bin/blossom_cuda   # this runs the fast CUDA GPU version
```

## Documentation

- Basic usage of the software and the description of the user interface is available in [HOWTO.md](./HOWTO.md).
- Some technical details about the code may be found in [src/README.md](./src/README.md).
- Doxygen-generated documentation of the source code can be found at https://molnsona.github.io/blossom/

### Quickstart
1. Click on the "plus" button on the bottom right side of the window
2. Choose **Open file** (the first button from the top) and open a file from the `demo_data/` directory
3. You can now add and delete landmarks using ctrl+mouse click, and drag them around.
4. Use the tools and settings available under the "plus" button to optimize the landmark positions and get a better visualization.

See the [HOWTO](./HOWTO.md) for more details and hints.
