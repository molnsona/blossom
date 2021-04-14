
set(HEADERS
    ui/application.h
    ui/imgui_config.h
    ui/imgui_utils.hpp
)
set(SOURCES
	${HEADERS}
    ui/application.cpp
)

add_executable(${MAIN_EXE_NAME}
    ${SOURCES}
	${HEADERS}
)

include_directories(.)

set_target_properties(${MAIN_EXE_NAME} PROPERTIES CXX_STANDARD 17)

if (MSVC)
    target_compile_options(${MAIN_EXE_NAME} BEFORE PUBLIC  /W4 /experimental:external /external:anglebrackets /external:W0)
else (MSVC)
    target_compile_options(${MAIN_EXE_NAME}  BEFORE PUBLIC  -Wall)
endif (MSVC)

# Magnum
find_package(Corrade REQUIRED Main)
find_package(Magnum REQUIRED GL Sdl2Application)
find_package(MagnumIntegration REQUIRED ImGui)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

target_link_libraries(${PROJECT_NAME} PRIVATE
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