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

#include "ui_load.h"

#include "parsers.h"
#include <exception>

UiLoader::UiLoader()
{
    opener.SetTitle("Open file");
    opener.SetTypeFilters(
      { ".fcs", ".tsv" }); // TODO take this from parsers, somehow
}

void
UiLoader::render(State state, ImGuiWindowFlags window_flags)
{
    opener.Display();

    if (opener.HasSelected()) {
        try {
            parse_generic(opener.GetSelected().string(), state.data);
        } catch (std::exception &e) {
            loading_error = e.what();
        }

        opener.ClearSelected();
    }

    if (!loading_error.empty()) {
        ImGui::Begin("Loading error", nullptr, window_flags);
        ImGui::Text(loading_error.c_str());
        if (ImGui::Button("OK"))
            loading_error = "";
        ImGui::End();
    }
}
