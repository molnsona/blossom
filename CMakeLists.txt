cmake_minimum_required(VERSION 3.11)

project(BlosSOM VERSION 0.0.2)

# Set the correct macro directory path
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})

# Setup glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "BUILD ACTIONS")

set(CMAKE_CXX_STANDARD 17)

if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W0> $<$<COMPILE_LANGUAGE:CXX>:/MP> $<$<COMPILE_LANGUAGE:CXX>:/DNOMINMAX>)
else (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-w> $<$<COMPILE_LANGUAGE:CXX>:-g>)
endif (MSVC)

find_package(OpenGL REQUIRED)

add_subdirectory(./glfw-3.3.8)
add_subdirectory(src)
#add_executable(blossom app.cpp)

#target_link_libraries(blossom glfw)



#target_link_libraries(blossom OpenGL::GL)