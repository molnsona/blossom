
set(SOURCES
    application.cpp
    embedsom.cpp
    embedsom_cuda.cpp
    fcs_parser.cpp
    graph_layout.cpp
    graph_renderer.cpp
    kmeans_landmark.cpp
    knn_edges.cpp
    landmark_model.cpp
    parsers.cpp
    scatter_renderer.cpp
    state.cpp
    trans_data.cpp
    tsv_parser.cpp
    ui_imgui.cpp
    ui_load.cpp
    ui_menu.cpp
    view.cpp
)

set(CUDA_SOURCES
    embedsom_cuda_topk.cu
    embedsom_cuda_projection.cu
)

# Regular (CPU-only target)
add_executable(blossom
    ${SOURCES}
)

include_directories(.)

set_target_properties(blossom PROPERTIES
    CXX_STANDARD 17
)

if (MSVC)
    target_compile_options(blossom BEFORE PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/experimental:external>
        $<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>
        $<$<COMPILE_LANGUAGE:CXX>:/external:W0>
    )
else (MSVC)
    target_compile_options(blossom BEFORE PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
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
    Threads::Threads
)

target_compile_definitions(blossom PRIVATE NO_CUDA)


# CUDA-accelerated target is compiled only if CUDA is available
if (BUILD_CUDA)
    find_package(CUDA)
    add_executable(blossom_cuda
        ${SOURCES}
        ${CUDA_SOURCES}
    )

    set_target_properties(blossom_cuda PROPERTIES
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86"
    )

    if (MSVC)
        target_compile_options(blossom_cuda BEFORE PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/experimental:external>
            $<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>
            $<$<COMPILE_LANGUAGE:CXX>:/external:W0>
        )
    else (MSVC)
        target_compile_options(blossom_cuda BEFORE PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
    endif (MSVC)
    
    target_include_directories(blossom_cuda PUBLIC include "${CUDA_INCLUDE_DIRS}")
    
    target_link_libraries(blossom_cuda PRIVATE
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

    install(TARGETS blossom blossom_cuda)

else (BUILD_CUDA)
    install(TARGETS blossom)
endif (BUILD_CUDA)
