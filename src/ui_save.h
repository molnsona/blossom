/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
 *                    Sona Molnarova
 *
 * BlosSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * BlosSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * BlosSOM. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef UI_SAVE_H
#define UI_SAVE_H

#include "imgui.h"
#include "vendor/imfilebrowser.h"

#include <array>
#include <string>

#include "state.h"

/**
 * @brief ImGUI handler for rendering the save file window.
 *
 */
struct UiSaver
{
    /**
     * @brief Types of data to be exported.
     *
     */
    enum Types
    {
        POINTS_HD,
        LAND_HD,
        POINTS_2D,
        LAND_2D,
        CLUSTERS,
        COUNT // Number of possible export types
    };

    /** Maximum size of the file name. */
    static constexpr int file_name_size = 128;

    /** If the save file window should be rendered. */
    bool show_window;
    /** ImGui file system dialog window handler.*/
    ImGui::FileBrowser saver;
    /** Error message of the saving file that will be shown in the error
     * window. */
    std::string saving_error;

    /** If all types of data should be exported. */
    bool all;
    /** Array of flags for each export data type indicating which data should be
     * exported and which not. */
    std::array<bool, UiSaver::Types::COUNT> data_flags;
    /** Names of the exported files for each export data type. */
    std::array<std::string, UiSaver::Types::COUNT> file_names;

    /**
     * @brief Initializes \p saver settings and initializes variables with
     * default values.
     *
     */
    UiSaver();
    /**
     * @brief Enables window to render.
     *
     */
    void show() { show_window = true; }

    /**
     * @brief Renders save file window, opens save file dialog window and calls
     * @ref save_data() if a directory was selected.
     *
     * @param app Application context.
     * @param window_flags Flags used for rendered window.
     */
    void render(State &state, ImGuiWindowFlags window_flags);

    /**
     * @brief Calls @ref write() for selected export data types.
     *
     * @param state Source of the exported data.
     * @param dir_name Name of the selected directory.
     */
    void save_data(const State &state, const std::string &dir_name) const;
    /**
     * @brief Writes given data into the file in the tsv format.
     *
     * @param type Type of the exported data.
     * @param state Source of the exported data.
     * @param dir_name Name of the selected directory.
     *
     * \exception std::domain_error Throws when the file cannot be opened for
     * writing.
     */
    void write(UiSaver::Types type,
               const State &state,
               const std::string &dir_name) const;
};

#endif
