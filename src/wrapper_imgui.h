
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

#ifndef WRAPPER_IMGUI_H
#define WRAPPER_IMGUI_H

#include "state.h"
#include "ui_menu.h"
#include "wrapper_glfw.h"

/**
 * @brief Wrapper of the ImGui.
 *
 * It abstracts the initialization and rendering of the UI.
 *
 */
class ImGuiWrapper
{
public:
    /**
     * @brief Initialize ImGui and load fonts.
     *
     * @param window
     * @return true
     * @return false
     */
    bool init(GLFWwindow *window);
    /**
     * @brief Render UI.
     *
     * @param callbacks
     * @param state
     */
    void render(CallbackValues callbacks, State &state);

    void destroy();

private:
    UiMenu menu;
};

#endif // WRAPPER_IMGUI_H
