
#ifndef UI_STORE_H
#define UI_STORE_H

#include "application.h"
#include <string>

struct uiStorer
{
    bool show_window;
    std::string loading_error;

    uiStorer();
    void show() { show_window = true; }
    void render(Application &app);
};

#endif
