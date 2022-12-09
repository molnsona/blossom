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

#include "ui_menu.h"

#include "imgui.h"
#include "vendor/IconsFontAwesome5.h"

#include "utils_imgui.hpp"

constexpr float WINDOW_PADDING = 100.0f;
constexpr float TOOLS_HEIGHT = 271.0f;
constexpr float WINDOW_WIDTH = 50.0f;

UiMenu::UiMenu()
  : show_menu(false)
{
}

static void
draw_menu_button(bool &show_menu, int fb_width, int fb_height)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoMove;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("Plus", nullptr, window_flags)) {
        ImGui::SetWindowPos(
          ImVec2(fb_width - WINDOW_PADDING, fb_height - WINDOW_PADDING));
        ImGui::SetWindowSize(ImVec2(WINDOW_WIDTH, WINDOW_WIDTH));

        if (ImGui::Button(ICON_FA_PLUS, ImVec2(50.75f, 50.75f))) {
            show_menu = !show_menu;
        }

        ImGui::End();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
UiMenu::render(int fb_width, int fb_height, State &state)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_AlwaysAutoResize;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);

    draw_menu_button(show_menu, fb_width, fb_height);
    if (show_menu)
        draw_menu_window(fb_width, fb_height, state);

    loader.render(state, window_flags);
    saver.render(state, window_flags);
    scaler.render(state, window_flags);
    training_set.render(state, window_flags);
    color_set.render(state, window_flags);

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
UiMenu::draw_menu_window(int fb_width, int fb_height, State &state)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoMove;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("Tools", &show_menu, window_flags)) {
        ImGui::SetWindowPos(ImVec2(fb_width - WINDOW_PADDING,
                                   fb_height - WINDOW_PADDING - TOOLS_HEIGHT));
        ImGui::SetWindowSize(ImVec2(WINDOW_WIDTH, TOOLS_HEIGHT));

        auto menu_entry = [&](auto icon, const char *label, auto &x) {
            if (ImGui::Button(icon, ImVec2(50.75f, 50.75f))) {
                x.show();
                show_menu = false;
            }
            tooltip(label);
        };

        menu_entry(ICON_FA_FOLDER_OPEN, "Open file", loader);
        menu_entry(ICON_FA_SAVE, "Save", saver);

        ImGui::Separator();

        menu_entry(ICON_FA_SLIDERS_H, "Scale data", scaler);
        menu_entry(ICON_FA_WRENCH, "Training settings", training_set);
        menu_entry(ICON_FA_PALETTE, "Color points", color_set);

        ImGui::End();
    }

    ImGui::PopStyleVar();
}
