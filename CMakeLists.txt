cmake_minimum_required(VERSION 3.11)

project(BlosSOM VERSION 0.0.2)

# Set project name
if (BUILD_CUDA)
    project(${PROJECT_NAME} LANGUAGES CXX CUDA)
else (BUILD_CUDA)
    project(${PROJECT_NAME} LANGUAGES CXX)
endif (BUILD_CUDA)

# Set the correct macro directory path
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})

# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "BUILD ACTIONS")

set(CMAKE_CXX_STANDARD 17)

if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W0> $<$<COMPILE_LANGUAGE:CXX>:/MP> $<$<COMPILE_LANGUAGE:CXX>:/DNOMINMAX>)
else (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-w> $<$<COMPILE_LANGUAGE:CXX>:-g>)
endif (MSVC)

# Check all the submodules
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/imgui-wrapper/imgui")
    message(FATAL_ERROR "The imgui submodule was not downloaded! GIT_SUBMODULE was turned off or failed.")
endif()

add_subdirectory(glad)
add_subdirectory(imgui-wrapper)

# data files
add_subdirectory(data)
add_compile_definitions(BLOSSOM_DATA_DIR="${BLOSSOM_DATA_DIR}")

add_subdirectory(src)
