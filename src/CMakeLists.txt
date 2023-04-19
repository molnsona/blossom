set(BLOSSOM_SOURCES
    batch_size_gen.cpp
    cluster_data.cpp
    color_data.cpp
    embedsom.cpp
    estimator.cpp
    fcs_parser.cpp
    graph_layout.cpp
    graph_renderer.cpp
    input_handler.cpp
    kmeans_landmark.cpp
    knn_edges.cpp
    landmark_model.cpp
    main.cpp
    parsers.cpp
    pnorm.cpp
    renderer.cpp
    scaled_data.cpp
    scatter_model.cpp
    scatter_renderer.cpp
    shader.cpp
    state.cpp
    texture_renderer.cpp
    trans_data.cpp
    tsne_layout.cpp
    tsv_parser.cpp
    ui_color.cpp
    ui_load.cpp
    ui_menu.cpp
    ui_save.cpp
    ui_scale.cpp
    ui_renderer.cpp
    ui_train.cpp
    wrapper_glad.cpp
    wrapper_glfw.cpp
    wrapper_imgui.cpp
)

set(CUDA_SOURCES
    embedsom_cuda_knn.cu
    embedsom_cuda_projection.cu
    embedsom_cuda.cpp
)

find_package(PkgConfig)
pkg_check_modules(glfw3 REQUIRED glfw3)
pkg_check_modules(glm REQUIRED glm)

set(BLOSSOM_LIBRARIES
    imgui    
    glad
    ${glfw3_LIBRARIES}
    ${glm_LIBRARIES}
)

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

target_link_libraries(blossom PRIVATE ${BLOSSOM_LIBRARIES})
target_include_directories(blossom PUBLIC include ${glfw3_INCLUDE_DIRS} ${glm_INCLUDE_DIRS})

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
    
    target_include_directories(blossom_cuda PUBLIC include ${CUDA_INCLUDE_DIRS} ${glfw3_INCLUDE_DIRS} ${glm_INCLUDE_DIRS})
    
    target_link_libraries(blossom_cuda PRIVATE
        ${BLOSSOM_LIBRARIES}
    )

    install(TARGETS blossom blossom_cuda)

else (BUILD_CUDA)
    install(TARGETS blossom)
endif (BUILD_CUDA)
