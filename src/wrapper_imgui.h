#ifndef WRAPPER_IMGUI_H
#define WRAPPER_IMGUI_H

#include "state.h"
#include "ui_menu.h"
#include "wrapper_glfw.h"

class ImGuiWrapper {
public:
    bool init(GLFWwindow* window);
    void render(CallbackValues callbacks, State state);
    void destroy();

private:
    UiMenu menu;
};

#endif // WRAPPER_IMGUI_H