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

#include "input_handler.h"
#include "renderer.h"
#include "state.h"
#include "timer.h"
#include "view.h"
#include "wrapper_glad.h"
#include "wrapper_glfw.h"
#include "wrapper_imgui.h"

#define DEBUG

#define MEASURE(name, method)\
    state.frame_stats.timer.tick();\
    state.frame_stats.constant_time +=\
        state.frame_stats.timer.frametime * 1000;\
    state.frame_stats.timer.tick();\
    method;\
    state.frame_stats.timer.tick();\
    state.frame_stats.constant_time +=\
        state.frame_stats.timer.frametime * 1000;\
    std::cout << name << state.frame_stats.timer.frametime * 1000 << std::endl;\
    state.frame_stats.timer.tick();

int
main()
{
    GlfwWrapper glfw;
    GladWrapper glad;
    ImGuiWrapper imgui;
    InputHandler input_handler;
    Renderer renderer;
    Timer timer;
    View view;
    State state;

    if (!glfw.init("BlosSOM", input_handler.input)) {
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

#ifdef DEBUG
        state.frame_stats.dt = timer.frametime * 1000;
#endif   

        state.frame_stats.timer.tick();
        input_handler.update(view, renderer, state);

        view.update(timer.frametime,
                    input_handler.input.fb_width,
                    input_handler.input.fb_height);
        
        state.update(timer.frametime,
                     renderer.get_vert_pressed(),
                     renderer.get_vert_ind());

        renderer.render(state, view);
        
        imgui.render(
          input_handler.input.fb_width, input_handler.input.fb_height, state);
       
        input_handler.reset();
        glfw.end_frame(state.frame_stats);

        state.frame_stats.timer.tick();
        state.frame_stats.constant_time += 
            state.frame_stats.timer.frametime * 1000;  
            
#ifdef DEBUG
    state.frame_stats.prev_const_time = state.frame_stats.constant_time;
#endif

        // // Because we want the frame to last ~17ms (~60 FPS).
        // float diff = 17.0f - state.frame_stats.constant_time;
        // Because we want the frame to last ~20ms (~50 FPS).        
        float diff = 20.0f - state.frame_stats.constant_time;
        state.frame_stats.est_time =
            diff < 0.0001f ? 1.0f : diff;   
        state.frame_stats.constant_time = 0.0f;           
    }

    imgui.destroy();
    glfw.destroy();
    return 0;
}
