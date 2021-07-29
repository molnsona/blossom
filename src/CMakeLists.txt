
set(HEADERS
    application.h
    #canvas.h
    data_model.h
    embedsom.h
    embedsom_cuda.h
    #graph.h
    graph_layout.h
    graph_renderer.h
    imgui_config.h
    landmark_model.h
    mouse_data.h
    pickable.hpp
    scatter_model.h
    scatter_renderer.h
    #scene_mngr.h
    state.h
    timer.h
    ui_imgui.h
    #utils.hpp
    view.h
)
set(SOURCES
    application.cpp
    #canvas.cpp
    data_model.cpp
    embedsom.cpp
    embedsom_cuda.cpp
    #graph.cpp
    graph_layout.cpp
    graph_renderer.cpp
    landmark_model.cpp
    scatter_model.cpp
    scatter_renderer.cpp
    #scene_mngr.cpp
    state.cpp
    ui_imgui.cpp
    view.cpp
)

set(CUDA_SOURCES
    embedsom_cuda_topk.cu
    #embedsom_cuda_projection.cu
)

add_executable(blossom
    ${SOURCES}
    ${HEADERS}
    ${CUDA_SOURCES}
)

include_directories(.)

set_target_properties(blossom PROPERTIES
    CXX_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "50;52;60;61"
)

if (MSVC)
    target_compile_options(blossom BEFORE PUBLIC  /W4 /experimental:external /external:anglebrackets /external:W0)
else (MSVC)
    target_compile_options(blossom BEFORE PUBLIC  -Wall)
endif (MSVC)

target_include_directories(blossom PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

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
