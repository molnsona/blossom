
#ifndef UI_TRAIN_H
#define UI_TRAIN_H

#include "application.h"
#include <string>

struct uiTrainingSettings
{
    bool show_window;
    std::string loading_error;

    uiTrainingSettings();
    void show() { show_window = true; }
    void render(Application &app, ImGuiWindowFlags window_flags);
};

#endif
