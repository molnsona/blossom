cmake_minimum_required(VERSION 3.11)

project(BlosSOM VERSION 0.0.2)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(./glfw-3.3.8)

add_executable(blossom app.cpp)

target_link_libraries(blossom glfw)

find_package(OpenGL REQUIRED)

target_link_libraries(blossom OpenGL::GL)