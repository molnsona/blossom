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

static void make_window(const char* name, const std::vector<float>& dts, 
const std::vector<size_t>& data)
{
    std::string input = "items,time ";

    bool open = true;
    ImGui::Begin(name, &open);
    for (size_t i = 0; i < 50; ++i)
    {
        auto di = data.size() <= i ? 0 : data[i];
        auto dti = dts.size() <= i ? 0 : dts[i];

        float items_per_sec = dti == 0 ? 0 : di / dti;
        input.append(std::to_string(di));
        input.append(","); 

        input.append(std::to_string(dti*1000));
        input.append(" ");  
    }
    char inputText[4096];
    strcpy(inputText, input.c_str());
    ImGui::InputText("##nameinput", inputText, 4096, ImGuiInputTextFlags_ReadOnly);

    ImGui::End();
}

static void debug_window(FrameStats& stats)
{
    make_window("trans debug", stats.trans_times, stats.trans_items);
    make_window("scatter debug", stats.scatter_times, stats.scatter_items);
    make_window("scaled debug", stats.scaled_times, stats.scaled_items);
    make_window("color debug", stats.color_times, stats.color_items);
}

#endif // #ifndef UTILS_IMGUI_HPP
