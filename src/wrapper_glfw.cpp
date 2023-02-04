#include "wrapper_glfw.h"

#include "imgui.h"
#include <glm/glm.hpp>

#include <iostream>

#define MEASURE(name, method)\
    fs.timer.tick();\
    fs.constant_time +=\
        fs.timer.frametime * 1000;\
    fs.timer.tick();\
    method;\
    fs.timer.tick();\
    fs.constant_time +=\
        fs.timer.frametime * 1000;\
    std::cout << name << fs.timer.frametime * 1000 << std::endl;\
    fs.timer.tick();


GlfwWrapper::GlfwWrapper() {}

bool
GlfwWrapper::init(const std::string &window_name, InputData &input)
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
    glfwSetWindowUserPointer(window, (void *)&input);

    return true;
}

bool
GlfwWrapper::window_should_close()
{
    return glfwWindowShouldClose(window);
}

void
GlfwWrapper::end_frame(FrameStats& fs)
{
    MEASURE("glFinish:  ", glFinish());
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
    InputData *input_inst = (InputData *)glfwGetWindowUserPointer(window);
    input_inst->fb_width = width;
    input_inst->fb_height = height;
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
    if (io.WantCaptureKeyboard)
        return;

    InputData *input_inst = (InputData *)glfwGetWindowUserPointer(window);
    input_inst->keyboard.key = key;
    input_inst->keyboard.action = action;
}

void
GlfwWrapper::scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    // Ignore callback if io is used by imgui window or gadget
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    InputData *input_inst = (InputData *)glfwGetWindowUserPointer(window);
    input_inst->mouse.xoffset = xoffset;
    input_inst->mouse.yoffset = yoffset;
}

void
GlfwWrapper::mouse_button_callback(GLFWwindow *window,
                                   int button,
                                   int action,
                                   int mods)
{
    // Ignore callback if io is used by imgui window or gadget
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    InputData *input_inst = (InputData *)glfwGetWindowUserPointer(window);
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    input_inst->mouse.pos = glm::vec2(xpos, ypos);
    input_inst->mouse.button = button;
    input_inst->mouse.action = action;
}

void
GlfwWrapper::cursor_position_callback(GLFWwindow *window,
                                      double xpos,
                                      double ypos)
{
    InputData *input_inst = (InputData *)glfwGetWindowUserPointer(window);
    input_inst->mouse.pos = glm::vec2(xpos, ypos);
}
