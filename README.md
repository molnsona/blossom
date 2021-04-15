# Interactive EmbedSOM research project

# Building

## Windows

## Unix-like systems
```sh
git submodule init
git submodule update

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE="Debug"
make
```

# Run
### !!! You have to change paths to fonts in src/ui/application.cpp `io.Fonts->AddFontFromFileTTF(...)`
``` sh
./Debug/bin/blosSOM
```

# Controls
Move: Arrows/Mouse drag

Zoom in: Esc/Mouse wheel

Zoom out: Space/Mouse wheel
