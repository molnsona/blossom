
#ifndef UI_COLOR_H
#define UI_COLOR_H

#include "application.h"
#include <string>

struct uiColorSettings
{
    bool show_window;
    std::string loading_error;

    uiColorSettings();
    void show() { show_window = true; }
    void render(Application &app, ImGuiWindowFlags window_flags);
};

#endif
