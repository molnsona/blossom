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

#include "ui_save.h"

#include "imgui_stdlib.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <exception>
#include <fstream>

UiSaver::UiSaver()
  : show_window(false)
  , saver(ImGuiFileBrowserFlags_SelectDirectory |
          ImGuiFileBrowserFlags_CreateNewDir)
  , all(false)
  , data_flags{ false, false, false, false, false }
  , file_names{ "points_hd.tsv",
                "landmarks_hd.tsv",
                "points_2d.tsv",
                "landmarks_2d.tsv",
                "clusters.tsv" }
{
    saver.SetTitle("Select directory");
}

void
UiSaver::render(State &state, ImGuiWindowFlags window_flags)
{
    if (!show_window)
        return;

    saver.Display();

    if (saver.HasSelected()) {
        try {
            save_data(state, saver.GetSelected().string());
        } catch (std::exception &e) {
            saving_error = e.what();
        }

        saver.ClearSelected();
    }

    if (!saving_error.empty()) {
        ImGui::Begin("Saving error", nullptr, window_flags);
        ImGui::Text(saving_error.c_str());
        if (ImGui::Button("OK"))
            saving_error = "";
        ImGui::End();
    }

    if (ImGui::Begin("Save##window", &show_window, window_flags)) {
        auto save_line = [&](const char *text, int type) {
            ImGui::Text(text);
            std::string checkbox_name =
              "##save_checkbox" + std::to_string(type);
            ImGui::Checkbox(checkbox_name.data(), &data_flags[type]);
            ImGui::SameLine();
            std::string inputtext_name =
              "file name##save" + std::to_string(type);
            ImGui::InputText(inputtext_name.data(), &file_names[type]);
        };

        save_line("Save hidim points:", UiSaver::Types::POINTS_HD);

        save_line("Save hidim landmarks:", UiSaver::Types::LAND_HD);

        save_line("Save 2D points:", UiSaver::Types::POINTS_2D);

        save_line("Save 2D landmarks:", UiSaver::Types::LAND_2D);

        save_line("Save clusters:", UiSaver::Types::CLUSTERS);
        ImGui::NewLine();

        if (ImGui::Checkbox("Save all", &all))
            if (all)
                std::fill(data_flags.begin(), data_flags.end(), true);
            else
                std::fill(data_flags.begin(), data_flags.end(), false);

        ImGui::NewLine();

        if (ImGui::Button("Save##button"))
            saver.Open();

        ImGui::End();
    }
}

void
UiSaver::save_data(const State &state, const std::string &dir_name) const
{
    if (data_flags[UiSaver::Types::POINTS_HD])
        write(UiSaver::Types::POINTS_HD, state, dir_name);

    if (data_flags[UiSaver::Types::LAND_HD])
        write(UiSaver::Types::LAND_HD, state, dir_name);

    if (data_flags[UiSaver::Types::POINTS_2D])
        write(UiSaver::Types::POINTS_2D, state, dir_name);

    if (data_flags[UiSaver::Types::LAND_2D])
        write(UiSaver::Types::LAND_2D, state, dir_name);

    if (data_flags[UiSaver::Types::CLUSTERS])
        write(UiSaver::Types::CLUSTERS, state, dir_name);
}

/**
 * @brief Writes multi-dimensional data into the file by a given handler.
 *
 * @param dim Dimension of the data.
 * @param data Multi-dimensional data to be written.
 * @param handle Handle to the opened file for writing.
 */
static void
write_data_float(size_t dim,
                 const std::vector<float> &data,
                 std::ofstream &handle)
{
    for (size_t i = 0; i < data.size(); i += dim) {
        for (size_t j = 0; j < dim - 1; ++j) {
            handle << data[i + j] << '\t';
        }
        handle << data[i + dim - 1] << '\n';
    }
};

/**
 * @brief Writes two-dimensional data into the file by a given handler.
 *
 * @param data Two-dimensional data to be written.
 * @param handle Handle to the opened file for writing.
 */
static void
write_data_2d(const std::vector<glm::vec2> &data, std::ofstream &handle)
{
    for (size_t i = 0; i < data.size(); ++i) {
        handle << data[i].x << '\t' << data[i].y << '\n';
    }
};

static void
write_clusters(std::vector<std::pair<const glm::vec3 *, int>> landmarks,
               const std::map<int, std::pair<glm::vec3, std::string>> &clusters,
               std::ofstream &handle)
{
    for (size_t i = 0; i < landmarks.size(); ++i) {
        auto id = landmarks[i].second;
        auto color = clusters.at(id).first;
        auto name = clusters.at(id).second;
        handle << i << '\t' << color.r << '\t' << color.g << '\t' << color.b
               << '\t' << name << '\n';
    }
}

void
UiSaver::write(Types type,
               const State &state,
               const std::string &dir_name) const
{
    std::string path = dir_name + "/" + file_names[type];
    std::ofstream handle(path, std::ios::out);
    if (!handle)
        throw std::domain_error("Can not open file");

    switch (type) {
        case UiSaver::Types::POINTS_HD:
            write_data_float(state.scaled.dim(), state.scaled.data, handle);
            break;
        case UiSaver::Types::LAND_HD:
            write_data_float(
              state.landmarks.d, state.landmarks.hidim_vertices, handle);
            break;
        case UiSaver::Types::POINTS_2D:
            write_data_2d(state.scatter.points, handle);
            break;
        case UiSaver::Types::LAND_2D:
            write_data_2d(state.landmarks.lodim_vertices, handle);
            break;
        case UiSaver::Types::CLUSTERS:
            write_clusters(
              state.colors.landmarks, state.colors.clustering.clusters, handle);
            break;
    }

    handle.close();
}
