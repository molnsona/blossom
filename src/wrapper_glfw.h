#ifndef WRAPPER_GLFW_H
#define WRAPPER_GLFW_H

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <string>

struct CallbackValues
{
    int fb_width = 800;
    int fb_height = 600;
};

class GlfwWrapper
{
public:
    GlfwWrapper();
    bool init(const std::string &window_name);
    bool window_should_close();
    void end_frame();
    void destroy();

    GLFWwindow *window;
    CallbackValues callbacks;

private:
    static void error_callback(int error, const char *description);
    static void framebuffer_size_callback(GLFWwindow *window,
                                          int width,
                                          int height);
    static void key_callback(GLFWwindow *window,
                             int key,
                             int scancode,
                             int action,
                             int mods);
};

#endif // WRAPPER_GLFW_H