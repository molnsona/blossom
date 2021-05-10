
set(HEADERS
    ui/application.h
    ui/imgui_config.h
    ui/imgui_utils.hpp
    utils.hpp
)
set(SOURCES
    ui/application.cpp
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
