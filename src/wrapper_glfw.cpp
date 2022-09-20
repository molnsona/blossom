#include "wrapper_glfw.h"

#include <iostream>

GlfwWrapper::GlfwWrapper() {}

bool
GlfwWrapper::init(const std::string &window_name)
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW." << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    window = glfwCreateWindow(800, 600, window_name.c_str(), NULL, NULL);
    if (!window) {
        std::cout << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Set the user pointer tied to the window used for
    // storing values in callbacks.
    glfwSetWindowUserPointer(window, (void *)this);

    return true;
}

bool
GlfwWrapper::window_should_close()
{
    return glfwWindowShouldClose(window);
}

void
GlfwWrapper::end_frame()
{
    glfwSwapBuffers(window);
    // Calls registered callbacks if any events were triggered
    glfwPollEvents();
}

void
GlfwWrapper::destroy()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

void
GlfwWrapper::error_callback(int error, const char *description)
{
    std::cerr << "Error: " << description << std::endl;
}

void
GlfwWrapper::framebuffer_size_callback(GLFWwindow *window,
                                       int width,
                                       int height)
{
    GlfwWrapper *glfw_inst = (GlfwWrapper *)glfwGetWindowUserPointer(window);
    glfw_inst->callbacks.fb_width = width;
    glfw_inst->callbacks.fb_height = height;
    glViewport(0, 0, width, height);
}

void
GlfwWrapper::key_callback(GLFWwindow *window,
                          int key,
                          int scancode,
                          int action,
                          int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}