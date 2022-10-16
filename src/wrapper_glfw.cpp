#include "wrapper_glfw.h"

#include "imgui.h"

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
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
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
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

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
    callbacks.reset();

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
    // Ignore callback if io is used by imgui window or gadget
    ImGuiIO &io = ImGui::GetIO();
    if(io.WantCaptureKeyboard) return;

    GlfwWrapper *glfw_inst = (GlfwWrapper *)glfwGetWindowUserPointer(window);
    glfw_inst->callbacks.key = key;
    glfw_inst->callbacks.key_action = action;
}

void GlfwWrapper::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    // Ignore callback if io is used by imgui window or gadget
    ImGuiIO &io = ImGui::GetIO();
    if(io.WantCaptureMouse) return;

    GlfwWrapper *glfw_inst = (GlfwWrapper *)glfwGetWindowUserPointer(window);
    glfw_inst->callbacks.xoffset = xoffset;
    glfw_inst->callbacks.yoffset = yoffset;
}

void GlfwWrapper::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    // Ignore callback if io is used by imgui window or gadget
    ImGuiIO &io = ImGui::GetIO();
    if(io.WantCaptureMouse) return;

    GlfwWrapper *glfw_inst = (GlfwWrapper *)glfwGetWindowUserPointer(window);
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    glfw_inst->callbacks.xpos = xpos;
    glfw_inst->callbacks.ypos = ypos;
    glfw_inst->callbacks.button = button;
    glfw_inst->callbacks.mouse_action = action;
}

void GlfwWrapper::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    GlfwWrapper *glfw_inst = (GlfwWrapper *)glfwGetWindowUserPointer(window);
    glfw_inst->callbacks.xpos = xpos;
    glfw_inst->callbacks.ypos = ypos;
}
