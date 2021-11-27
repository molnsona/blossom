
#ifndef UI_SCALE_H
#define UI_SCALE_H

#include "application.h"
#include <string>

struct uiScaler
{
    bool show_window;
    std::string loading_error;

    static constexpr float slider_width = 150.0f;

    uiScaler();
    void show() { show_window = true; }
    void render(Application &app, ImGuiWindowFlags window_flags);
};

#endif
