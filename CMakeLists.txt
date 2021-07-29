cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(BlosSOM VERSION 0.0.1)

# Set project name
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# Set the correct macro directory path
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})

# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "BUILD ACTIONS")

# Set C++17
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
  set(CMAKE_CXX_STANDARD 17)
endif(MSVC)

if (MSVC)
  add_compile_options(/W0 /MP)
else (MSVC)
  add_compile_options(-w -g)
endif (MSVC)

# setup magnum
set(BUILD_STATIC ON CACHE BOOL "" FORCE)
set(WITH_SDL2APPLICATION ON CACHE BOOL "" FORCE)
set(WITH_IMGUI ON CACHE BOOL "" FORCE)
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

# add magnum
add_subdirectory(corrade EXCLUDE_FROM_ALL)
add_subdirectory(magnum EXCLUDE_FROM_ALL)
add_subdirectory(magnum-integration EXCLUDE_FROM_ALL)

# find magnum libraries
find_package(Corrade REQUIRED Main)
find_package(Magnum REQUIRED GL Sdl2Application)
find_package(MagnumIntegration REQUIRED ImGui)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

# data files
add_subdirectory(data)
add_compile_definitions(BLOSSOM_DATA_DIR="${BLOSSOM_DATA_DIR}")

# program sources
add_subdirectory(src)

