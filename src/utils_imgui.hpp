/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Sona Molnarova
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

#ifndef UTILS_IMGUI_HPP
#define UTILS_IMGUI_HPP

#include "frame_stats.h"

#include <imgui.h>
#include <string>
#include <vector>

/**
 * @brief ImGUI wrapper for setting tooltip.
 *
 * Call this after the widget you want to use the tooltip on.
 *
 * @param text Tooltip text.
 */
static void
tooltip(const char *text)
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2, 2));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8f);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(text);
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

/**
 * @brief ImGUI wrapper for reset button.
 *
 * It does not reset anything. It just creates button and returns if the button
 * was pressed. It is on the user to reset the data after the button was
 * pressed.
 *
 * @return true If the button was pressed.
 * @return false If the button was not pressed.
 */
static bool
reset_button()
{
    ImGui::SameLine();
    float width = ImGui::GetWindowContentRegionWidth() - 70.0f;
    ImGui::Indent(width);
    bool res = ImGui::Button("Reset data");
    ImGui::Unindent(width);
    return res;
}

static void
add2window(const char *name, float t, size_t n)
{
    // std::string input = "N,T ";

    // input.append(std::to_string(n));
    // input.append(",");
    // input.append(std::to_string(t));
    // input.append(" ");

    // char inputText[4096];
    // strcpy(inputText, input.c_str());
    // ImGui::InputText(
    //   name, inputText, 4096, ImGuiInputTextFlags_ReadOnly);
   
   ImGui::Text("%zu, %f\t\t\t%s", n, t, name);
}

static void
debug_windows(FrameStats &fs)
{
    bool open = true;
    ImGui::Begin("debug estimated batch sizes", &open);
    add2window("trans", fs.trans_t, fs.trans_n);
    add2window("scaled", fs.scaled_t, fs.scaled_n);
    add2window("embedsom", fs.embedsom_t, fs.embedsom_n);
    add2window("color", fs.color_t, fs.color_n);
    ImGui::End();

    ImGui::Begin("debug times", &open);
    ImGui::Text("dt:\t\t\t\t\t\t\t%f", fs.dt);
    ImGui::Text("\t-constant:\t\t%f", fs.prev_const_time);
    ImGui::Text("\t\t-glFinish:\t\t%f", fs.gl_finish_time);
    ImGui::Text("\t-estimated:\t\t%f", fs.est_time);
    ImGui::Text("\t\t-trans:\t\t\t\t%f", fs.trans_t);
    ImGui::Text("\t\t-scaled:\t\t\t%f", fs.scaled_t);
    ImGui::Text("\t\t-color:\t\t\t\t%f", fs.color_t);
    ImGui::Text("\t\t-embedsom:\t%f", fs.embedsom_t);
    ImGui::End();
}

#endif // #ifndef UTILS_IMGUI_HPP
