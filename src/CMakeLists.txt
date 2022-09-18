set(BLOSSOM_SOURCES
    app.cpp
)

set(BLOSSOM_LIBRARIES
    imgui
    glfw
    glad
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