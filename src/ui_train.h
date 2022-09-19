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

#ifndef UI_TRAIN_H
#define UI_TRAIN_H

#include "imgui.h"

#include "state.h"

/**
 * @brief ImGUI handler for rendering the training settings window.
 *
 */
struct UiTrainingSettings
{
    /** If the training settings window should be rendered. */
    bool show_window;

    UiTrainingSettings();
    /**
     * @brief Enables window to render.
     *
     */
    void show() { show_window = true; }
    /**
     * @brief Renders window with corresponding training settings widgets.
     *
     * @param app Application context.
     * @param window_flags Flags used for rendered window.
     */
    void render(State& state, ImGuiWindowFlags window_flags);
};

#endif
