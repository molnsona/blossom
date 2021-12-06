
#ifndef UI_STORE_H
#define UI_STORE_H

#include "application.h"
#include "exporter.h"
#include "extern/imfilebrowser.h"

#include <string>

struct uiExporter
{
    static constexpr int file_name_size = 128;

    bool show_window;
    ImGui::FileBrowser opener;
    std::string saving_error;

    Exporter exporter;

    uiExporter();
    void show() { show_window = true; }
    void render(Application &app, ImGuiWindowFlags window_flags);
};

#endif
