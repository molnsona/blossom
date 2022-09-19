#ifndef WRAPPER_IMGUI_H
#define WRAPPER_IMGUI_H

#include <GLFW/glfw3.h>

class ImGuiWrapper {
public:
    bool init(GLFWwindow* window);
    void compose_frame();
    void render();
    void destroy();
};

#endif // WRAPPER_IMGUI_H