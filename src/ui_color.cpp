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

#include "application.h"
#include "utils_imgui.hpp"
#include "vendor/colormap/palettes.hpp"

uiColorSettings::uiColorSettings()
  : show_window(false)
{}

void
uiColorSettings::render(Application &app, ImGuiWindowFlags window_flags)
{
    if (!show_window)
        return;

    auto column_combo = [&](const std::string &combo_name, int &column_ind) {
        ImGui::Text("Column for colors:");
        auto dim = app.state.data.names.size();
        if (!dim) {
            ImGui::Text("No columns are present.");
            return;
        }

        if (ImGui::BeginCombo(combo_name.data(),
                              app.state.data.names[column_ind].c_str())) {
            bool ret = false;
            for (size_t i = 0; i < dim; ++i) {
                bool is_selected = (int)i == column_ind;
                if (ImGui::Selectable(app.state.data.names[i].c_str(),
                                      &is_selected)) {
                    column_ind = i;
                    ret = true;
                }
            }

            if (ret)
                app.state.colors.touch_config();
            ImGui::EndCombo();
        }
    };

    if (ImGui::Begin("Color", &show_window, window_flags)) {
        ImGui::Text("Style of coloring:");

        if (reset_button()) {
            app.state.colors.reset();
        }

        if (ImGui::RadioButton("Expression",
                               &app.state.colors.coloring,
                               (int)ColorData::Coloring::EXPR))
            app.state.colors.touch_config();

        if (ImGui::RadioButton("Discretized clusters",
                               &app.state.colors.coloring,
                               (int)ColorData::Coloring::CLUSTER))
            app.state.colors.touch_config();

        if (ImGui::SliderFloat("Alpha##color",
                               &app.state.colors.alpha,
                               0.0f,
                               1.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp))
            app.state.colors.touch_config();

        ImGui::Separator();

        switch (app.state.colors.coloring) {
            case (int)ColorData::Coloring::EXPR: {
                column_combo("##columnsexpr", app.state.colors.expr_col);

                ImGui::Text("Color palette:");
                if (ImGui::BeginCombo("##palettes",
                                      app.state.colors.col_palette.c_str())) {
                    bool ret = false;
                    for (auto &[name, _] : colormap::palettes) {
                        bool is_selected = name == app.state.colors.col_palette;
                        if (ImGui::Selectable(name.c_str(), &is_selected)) {
                            app.state.colors.col_palette = name;
                            ret = true;
                        }
                    }

                    if (ret)
                        app.state.colors.touch_config();
                    ImGui::EndCombo();
                }

                if (ImGui::Checkbox("Reverse color palette",
                                    &app.state.colors.reverse))
                    app.state.colors.touch_config();
            } break;
            case int(ColorData::Coloring::CLUSTER):
                column_combo("##columnscluster", app.state.colors.cluster_col);

                if (ImGui::SliderInt(
                      "Cluster count", &app.state.colors.cluster_cnt, 1, 50))
                    app.state.colors.touch_config();

                break;
        }

        ImGui::End();
    }
}
