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

#ifndef UI_SCALE_H
#define UI_SCALE_H

#include "imgui.h"

#include "state.h"

/**
 * @brief ImGUI handler for rendering the scale&transform window.
 *
 */
struct UiScaler
{
    /** If the scale&transform window should be rendered. */
    bool show_window;

    /** Width of the sliders in the table. */
    static constexpr float slider_width = 150.0f;

    UiScaler();
    /**
     * @brief Enables window to render.
     *
     */
    void show() { show_window = true; }
    /**
     * @brief Renders window with corresponding scale&transform widgets.
     *
     * @param app Application context.
     * @param window_flags Flags used for rendered window.
     */
    void render(State &state, ImGuiWindowFlags window_flags);
};

#endif
