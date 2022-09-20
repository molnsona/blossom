
#include "wrapper_glad.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/gl.h>

#include <iostream>

bool
GladWrapper::init()
{
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD." << std::endl;
        return false;
    }

    return true;
}