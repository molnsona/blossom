
#ifndef UI_SAVE_H
#define UI_SAVE_H

#include "application.h"
#include "extern/imfilebrowser.h"

#include <array>
#include <string>

struct uiSaver
{
    enum Types
    {
        POINTS_HD,
        LAND_HD,
        POINTS_2D,
        LAND_2D,
        COUNT // Number of possible export types
    };

    static constexpr int file_name_size = 128;

    bool show_window;
    ImGui::FileBrowser opener;
    std::string saving_error;

    bool all;
    std::array<bool, uiSaver::Types::COUNT> data_flags;
    std::array<std::string, uiSaver::Types::COUNT> file_names;

    uiSaver();
    void show() { show_window = true; }
    void render(Application &app, ImGuiWindowFlags window_flags);

    void save_data(const State &state, const std::string &dir_name);
    void write(uiSaver::Types type,
               const State &state,
               const std::string &dir_name);
};

#endif
