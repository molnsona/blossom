# Interactive EmbedSOM research project

# Building

## Windows

## Unix-like systems
```sh
git clone https://github.com/mosra/corrade.git
git clone https://github.com/mosra/magnum.git
git clone https://github.com/mosra/magnum-integration.git
git clone https://github.com/ocornut/imgui.git

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