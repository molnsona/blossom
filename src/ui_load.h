
#ifndef UI_LOAD_H
#define UI_LOAD_H

#include "application.h"
#include "vendor/imfilebrowser.h"
#include <string>

struct uiLoader
{
    ImGui::FileBrowser opener;
    std::string loading_error;

    uiLoader();
    void show() { opener.Open(); }
    void render(Application &app);
};

#endif
