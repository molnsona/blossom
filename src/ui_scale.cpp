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

uiScaler::uiScaler()
  : show_window(false)
{}

void
uiScaler::render(Application &app, ImGuiWindowFlags window_flags)
{
    if (!show_window)
        return;

    if (app.state.data.names.size() != app.state.trans.dim()) {
        ImGui::Begin("Data error", nullptr, window_flags);
        ImGui::Text("Data has different dimension than transformed data.");
        if (ImGui::Button("OK")) {
            show_window = false;
        }
        ImGui::End();
        return;
    }

    if (app.state.data.names.size() != app.state.scaled.dim()) {
        ImGui::Begin("Data error", nullptr, window_flags);
        ImGui::Text("Data has different dimension than scaled data.");
        if (ImGui::Button("OK")) {
            show_window = false;
        }
        ImGui::End();
        return;
    }

    auto dim = app.state.trans.dim();

    if (ImGui::Begin("Scale", &show_window, window_flags)) {
        if (reset_button()) {
            app.state.trans.reset();
            app.state.scaled.reset();
        }

        if (!dim) {
            ImGui::Text("No columns were detected.");
            ImGui::End();
            return;
        }

        ImGui::BeginTable("##tabletrans", 6);
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        ImGui::Text("asinh");
        ImGui::TableNextColumn();
        ImGui::Text("asinh cofactor");
        ImGui::TableNextColumn();
        ImGui::Text("affine adjust");
        ImGui::TableNextColumn();
        ImGui::Text("scale");
        ImGui::TableNextColumn();
        ImGui::Text("sdev");

        for (size_t i = 0; i < dim; ++i) {
            ImGui::TableNextColumn();
            ImGui::Text(app.state.data.names[i].c_str());

            ImGui::TableNextColumn();
            std::string name = "##asinh" + std::to_string(i);
            if (ImGui::Checkbox(name.data(), &app.state.trans.config[i].asinh))
                app.state.trans.touch_config();

            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(slider_width);
            name = "##asinh_cofactor" + std::to_string(i);
            if (ImGui::SliderFloat(name.data(),
                                   &app.state.trans.config[i].asinh_cofactor,
                                   0.1f,
                                   1000.0f,
                                   "%.3f",
                                   ImGuiSliderFlags_AlwaysClamp))
                app.state.trans.touch_config();

            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(slider_width);
            name = "##affine_adjust" + std::to_string(i);
            if (ImGui::SliderFloat(name.data(),
                                   &app.state.trans.config[i].affine_adjust,
                                   -3.0f,
                                   3.0f,
                                   "%.3f",
                                   ImGuiSliderFlags_AlwaysClamp))
                app.state.trans.touch_config();

            ImGui::TableNextColumn();
            name = "##scale" + std::to_string(i);
            if (ImGui::Checkbox(name.data(), &app.state.scaled.config[i].scale))
                app.state.scaled.touch_config();

            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(slider_width);
            name = "##sdev" + std::to_string(i);
            if (ImGui::SliderFloat(name.data(),
                                   &app.state.scaled.config[i].sdev,
                                   0.1f,
                                   5.0f,
                                   "%.3f",
                                   ImGuiSliderFlags_AlwaysClamp))
                app.state.scaled.touch_config();
        }

        ImGui::EndTable();
        ImGui::End();
    }
}
