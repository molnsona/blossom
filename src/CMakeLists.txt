
set(HEADERS
    application.h
    canvas.h
    data_model.h
    graph.h
    graph_layout.h
    graph_model.h
    graph_renderer.h
    imgui_config.h
    pickable.hpp
    scatter_model.h
    scatter_renderer.h
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
    data_model.cpp
    graph.cpp
    graph_layout.cpp
    graph_model.cpp
    graph_renderer.cpp
    scene_mngr.cpp
    scatter_model.cpp
    scatter_renderer.cpp
    state.cpp
    ui_imgui.cpp
    view.cpp
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
