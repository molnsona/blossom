#ifndef WRAPPER_IMGUI_H
#define WRAPPER_IMGUI_H

#include "ui_menu.h"
#include "wrapper_glfw.h"

class ImGuiWrapper {
public:
    bool init(GLFWwindow* window);
    void render(CallbackValues callbacks);
    void destroy();

private:
    UiMenu menu;
};

#endif // WRAPPER_IMGUI_H