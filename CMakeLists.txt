cmake_minimum_required(VERSION 3.4...3.8)

set(PROJECT_NAME blosSOM)
set(MAIN_EXE_NAME blosSOM)

# Set project name
project(${PROJECT_NAME})

# Set version
set(CMAKE_PROJECT_VERSION_MAJOR 0)
set(CMAKE_PROJECT_VERSION_MINOR 0)
set(CMAKE_PROJECT_VERSION_PATCH 1)
set(CMAKE_PROJECT_VERSION ${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}.${CMAKE_PROJECT_VERSION_PATCH})

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

# Install dir
set(CMAKE_INSTALL_PREFIX "./dist")


# Add Corrade as a subproject
add_subdirectory(corrade EXCLUDE_FROM_ALL)

# Add Magnum as a subproject, enable Sdl2Application
set(WITH_SDL2APPLICATION ON CACHE BOOL "" FORCE)
set(WITH_IMGUI ON CACHE BOOL "" FORCE)
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

add_subdirectory(magnum EXCLUDE_FROM_ALL)
add_subdirectory(magnum-integration EXCLUDE_FROM_ALL)

add_subdirectory(src)