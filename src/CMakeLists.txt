set(BLOSSOM_SOURCES
    color_data.cpp
    embedsom.cpp
    fcs_parser.cpp
    graph_layout.cpp
    kmeans_landmark.cpp
    knn_edges.cpp
    landmark_model.cpp
    main.cpp
    parsers.cpp
    pnorm.cpp
    renderer.cpp
    scaled_data.cpp
    scatter_model.cpp
    shader.cpp
    state.cpp
    trans_data.cpp
    tsne_layout.cpp
    tsv_parser.cpp
    ui_color.cpp
    ui_load.cpp
    ui_menu.cpp
    ui_save.cpp
    ui_scale.cpp
    ui_train.cpp
    wrapper_glad.cpp
    wrapper_glfw.cpp
    wrapper_imgui.cpp
)

set(BLOSSOM_LIBRARIES
    imgui
    glfw
    glad
    glm
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

target_link_libraries(blossom PRIVATE
    ${BLOSSOM_LIBRARIES}
)

install(TARGETS blossom)