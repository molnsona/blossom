
set(HEADERS
    application.h
    state.h
    pickable.hpp
    canvas.h
    graph.h
    scene_mngr.h
    simulation.h
    imgui_config.h
    imgui_utils.hpp
    ui_imgui.h
    utils.hpp
)
set(SOURCES
    application.cpp
    state.cpp
    canvas.cpp
    graph.cpp
    scene_mngr.cpp
    simulation.cpp
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
