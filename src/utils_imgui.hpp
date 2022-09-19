#ifndef UTILS_IMGUI_HPP
#define UTILS_IMGUI_HPP

#include <imgui.h>

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

#endif // #ifndef UTILS_IMGUI_HPP
