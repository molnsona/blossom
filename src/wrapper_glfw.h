#ifndef WRAPPER_GLFW_H
#define WRAPPER_GLFW_H

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <string>

struct CallbackValues
{
    CallbackValues() {reset();}
    
    int fb_width = 800;
    int fb_height = 600;

    int key;
    int key_action;

    double xoffset;
    double yoffset;

    // Raw mouse cursor position([0,0] in the upper left corner).
    // Have to convert it to coordinates with [0,0] in
    // the middle of the screen.
    double xpos;
    double ypos;
    int button;
    int mouse_action;
    bool left_click = false;

    void reset()
    {
        key = 0;
        xoffset = 0;
        yoffset = 0;
        button = -1;
    }
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
    static void scroll_callback(GLFWwindow *window,
                                double xoffset,
                                double yoffset);
    static void mouse_button_callback(GLFWwindow *window,
                                      int button,
                                      int action,
                                      int mods);
    static void cursor_position_callback(GLFWwindow *window,
                                         double xpos,
                                         double ypos);
};

#endif // WRAPPER_GLFW_H
