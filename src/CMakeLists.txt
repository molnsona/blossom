
set(HEADERS
    application.h
    canvas.h
    graph.h
    graph_layout.h
    imgui_config.h
    imgui_utils.hpp
    pickable.hpp
    scene_mngr.h
    state.h
    timer.h
    ui_imgui.h
    utils.hpp
    view.h
)
set(SOURCES
    application.cpp
    canvas.cpp
    graph.cpp
    graph_layout.cpp
    scene_mngr.cpp
    state.cpp
    ui_imgui.cpp
)

add_executable(blossom
    ${SOURCES}
    ${HEADERS}
)

include_directories(.)

set_target_properties(blossom PROPERTIES CXX_STANDARD 17)

if (MSVC)
    target_compile_options(blossom BEFORE PUBLIC  /W4 /experimental:external /external:anglebrackets /external:W0)
else (MSVC)
    target_compile_options(blossom  BEFORE PUBLIC  -Wall)
endif (MSVC)

target_link_libraries(blossom PRIVATE
    Corrade::Main
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::MeshTools
    Magnum::Primitives
    Magnum::SceneGraph
    Magnum::Shaders
    Magnum::Trade
    MagnumIntegration::ImGui
)

install(TARGETS blossom)
