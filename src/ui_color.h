/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
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

#ifndef UI_COLOR_H
#define UI_COLOR_H

#include "state.h"

#include "imgui.h"

/**
 * @brief ImGUI handler for rendering the color settings window.
 *
 */
struct UiColorSettings
{
    /** If the color settings window should be rendered. */
    bool show_window;

    UiColorSettings();
    /**
     * @brief Enables window to render.
     *
     */
    void show() { show_window = true; }
    /**
     * @brief Renders window with corresponding color settings widgets.
     *
     * @param app Application context.
     * @param window_flags Flags used for rendered window.
     */
    void render(State &state, ImGuiWindowFlags window_flags);
};

#endif
