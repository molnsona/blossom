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

# Download all submodules
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# Check all the submodules
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/glfw/CMakeLists.txt")
    message(FATAL_ERROR "The glfw submodule was not downloaded! GIT_SUBMODULE was turned off or failed.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/imgui-wrapper/imgui")
    message(FATAL_ERROR "The imgui submodule was not downloaded! GIT_SUBMODULE was turned off or failed.")
endif()

add_subdirectory(glfw)
add_subdirectory(glad)
add_subdirectory(imgui-wrapper)

add_subdirectory(src)