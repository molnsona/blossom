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

#include "ui_color.h"

#include "imgui_stdlib.h"
#include "vendor/IconsFontAwesome5.h"
#include "vendor/colormap/palettes.hpp"

#include "utils_imgui.hpp"

UiColorSettings::UiColorSettings()
  : show_window(false)
{
}

void
UiColorSettings::render(State &state, ImGuiWindowFlags window_flags)
{
    if (!show_window) {
        state.colors.clustering.active_cluster = -1;
        return;
    }

    auto column_combo = [&](const std::string &combo_name, int &column_ind) {
        ImGui::Text("Column for colors:");
        auto dim = state.data.names.size();
        if (!dim) {
            ImGui::Text("No columns are present.");
            return;
        }

        if (ImGui::BeginCombo(combo_name.data(),
                              state.data.names[column_ind].c_str())) {
            bool ret = false;
            for (size_t i = 0; i < dim; ++i) {
                bool is_selected = (int)i == column_ind;
                if (ImGui::Selectable(state.data.names[i].c_str(),
                                      &is_selected)) {
                    column_ind = i;
                    ret = true;
                }
            }

            if (ret)
                state.colors.touch_config();
            ImGui::EndCombo();
        }
    };

    if (ImGui::Begin("Color", &show_window, window_flags)) {
        ImGui::Text("Style of coloring:");

        if (reset_button()) {
            state.colors.reset();
        }

        if (ImGui::RadioButton("Expression",
                               &state.colors.coloring,
                               (int)ColorData::Coloring::EXPR))
            state.colors.touch_config();

        if (ImGui::RadioButton("Discretized clusters",
                               &state.colors.coloring,
                               (int)ColorData::Coloring::CLUSTER))
            state.colors.touch_config();

        if (ImGui::RadioButton("Brushing",
                               &state.colors.coloring,
                               (int)ColorData::Coloring::BRUSHING))
            state.colors.touch_config();

        if (ImGui::SliderFloat("Alpha##color",
                               &state.colors.alpha,
                               0.0f,
                               1.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp))
            state.colors.touch_config();

        ImGui::Separator();

        switch (state.colors.coloring) {
            case (int)ColorData::Coloring::EXPR: {
                column_combo("##columnsexpr", state.colors.expr_col);

                ImGui::Text("Color palette:");
                if (ImGui::BeginCombo("##palettes",
                                      state.colors.col_palette.c_str())) {
                    bool ret = false;
                    for (auto &[name, _] : colormap::palettes) {
                        bool is_selected = name == state.colors.col_palette;
                        if (ImGui::Selectable(name.c_str(), &is_selected)) {
                            state.colors.col_palette = name;
                            ret = true;
                        }
                    }

                    if (ret)
                        state.colors.touch_config();
                    ImGui::EndCombo();
                }

                if (ImGui::Checkbox("Reverse color palette",
                                    &state.colors.reverse))
                    state.colors.touch_config();
            } break;
            case int(ColorData::Coloring::CLUSTER):
                column_combo("##columnscluster",
                             state.colors.clustering.cluster_col);

                if (ImGui::SliderInt("Cluster count",
                                     &state.colors.clustering.cluster_cnt,
                                     1,
                                     50))
                    state.colors.touch_config();

                break;
            case int(ColorData::Coloring::BRUSHING):
                auto &clustering = state.colors.clustering;
                auto cluster_size = clustering.clusters.size();
                if (cluster_size != 0) {
                    ImGui::SliderFloat("Radius##mouseradiuscircle",
                                       &state.colors.clustering.radius_size,
                                       1.0f,
                                       100.0f,
                                       "%.3f",
                                       ImGuiSliderFlags_AlwaysClamp);

                    ImGui::RadioButton("None", &clustering.active_cluster, -1);
                }

                for (auto iter = clustering.clusters.begin();
                     iter != clustering.clusters.end();) {
                    int i = iter->first;

                    // Skip none cluster, it is there only for
                    // exporting the data.
                    if(iter->first == -1) {
                        ++iter; 
                        continue;
                    }

                    ImGui::RadioButton(
                      ("##BrushingCluster" + std::to_string(i)).data(),
                      &clustering.active_cluster,
                      int(i));

                    ImGui::SameLine();
                    ImGui::ColorEdit3(
                      ("##ClusterColor" + std::to_string(i)).data(),
                      (float *)&clustering.clusters[i].first,
                      ImGuiColorEditFlags_NoInputs |
                        ImGuiColorEditFlags_NoLabel |
                        ImGuiColorEditFlags_AlphaPreview);
                    ImGui::SameLine();
                    ImGui::InputText(
                      ("##ClusterName" + std::to_string(i)).data(),
                      &clustering.clusters[i].second);
                    ImGui::SameLine();
                    if (ImGui::Button(
                          (ICON_FA_TIMES "##" + std::to_string(i)).data())) {
                        clustering.active_cluster = -1;
                        iter = clustering.clusters.erase(iter);
                        state.colors.reset_landmark_color(i);
                    } else {
                        ++iter;
                    }
                    tooltip("Remove cluster");
                }

                if (ImGui::Button("+ Add cluster")) {
                    clustering.add_cluster();
                }
                break;
        }

        ImGui::End();
    }
}
