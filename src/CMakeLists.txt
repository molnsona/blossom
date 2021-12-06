
set(BLOSSOM_SOURCES
    application.cpp
    color_data.cpp
    embedsom.cpp
    fcs_parser.cpp
    graph_layout.cpp
    graph_renderer.cpp
    kmeans_landmark.cpp
    knn_edges.cpp
    landmark_model.cpp
    parsers.cpp
    pnorm.cpp
    scaled_data.cpp
    scatter_model.cpp
    scatter_renderer.cpp
    state.cpp
    trans_data.cpp
    tsne_layout.cpp
    tsv_parser.cpp
    ui_color.cpp
    ui_imgui.cpp
    ui_load.cpp
    ui_menu.cpp
    ui_save.cpp
    ui_scale.cpp
    ui_train.cpp
    view.cpp
)

set(CUDA_SOURCES
    embedsom_cuda_knn.cu
    embedsom_cuda_projection.cu
    embedsom_cuda.cpp
)

set(BLOSSOM_LIBRARIES
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
    
# Regular (CPU-only target)
add_executable(blossom
    ${BLOSSOM_SOURCES}
)

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
    ${BLOSSOM_LIBRARIES}
)

# CUDA-accelerated target is compiled only if CUDA is available
if (BUILD_CUDA)
    find_package(CUDA)
    add_executable(blossom_cuda
        ${BLOSSOM_SOURCES}
        ${CUDA_SOURCES}
    )

    target_compile_definitions(blossom_cuda PRIVATE ENABLE_CUDA)

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
        ${BLOSSOM_LIBRARIES}
    )

    install(TARGETS blossom blossom_cuda)

else (BUILD_CUDA)
    install(TARGETS blossom)
endif (BUILD_CUDA)
