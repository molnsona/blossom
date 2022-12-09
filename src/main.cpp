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

#include <iostream>

#include "renderer.h"
#include "state.h"
#include "timer.h"
#include "view.h"
#include "wrapper_glad.h"
#include "wrapper_glfw.h"
#include "wrapper_imgui.h"

int
main()
{
    GlfwWrapper glfw;
    GladWrapper glad;
    ImGuiWrapper imgui;
    Renderer renderer;
    Timer timer;
    View view;
    State state;

    if (!glfw.init("BlosSOM")) {
        std::cout << "GLFW initialization failed." << std::endl;
        return -1;
    }

    if (!glad.init()) {
        std::cout << "GLAD initialization failed." << std::endl;
        glfw.destroy();
        return -1;
    }

    if (!imgui.init(glfw.window)) {
        std::cout << "Dear ImGui initialization failed." << std::endl;
        glfw.destroy();
        return -1;
    }

    if (!renderer.init()) {
        std::cout << "Renderer initialization failed." << std::endl;
        imgui.destroy();
        glfw.destroy();
        return -1;
    }

    while (!glfw.window_should_close()) {
        timer.tick();

        view.update(timer.frametime, glfw.callbacks);
        state.update(timer.frametime);

        renderer.update(state, view, glfw.callbacks);

        imgui.render(glfw.callbacks, state);

        glfw.end_frame();
    }

    imgui.destroy();
    glfw.destroy();
    return 0;
}
