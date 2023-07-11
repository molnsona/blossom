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

#ifndef WRAPPER_GLFW_H
#define WRAPPER_GLFW_H

#include "input_data.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <string>

#include "frame_stats.h"

/**
 * @brief Wrapper of the Glfw library.
 *
 * It abstracts window creation, deletition and handles callbacks.
 *
 */
class GlfwWrapper
{
public:
    GlfwWrapper();
    ~GlfwWrapper();
    bool init(const std::string &window_name, InputData &input);
    bool window_should_close();
    void end_frame(FrameStats &fs);
    void destroy();

    GLFWwindow *window;

private:
    static void error_callback(int error, const char *description);
    static void framebuffer_size_callback(GLFWwindow *window,
                                          int width,
                                          int height);
    static void key_callback(GLFWwindow *window,
                             int key,
                             int scancode,
                             int action,
                             int mods);
    static void scroll_callback(GLFWwindow *window,
                                double xoffset,
                                double yoffset);
    static void mouse_button_callback(GLFWwindow *window,
                                      int button,
                                      int action,
                                      int mods);
    static void cursor_position_callback(GLFWwindow *window,
                                         double xpos,
                                         double ypos);
};

#endif // WRAPPER_GLFW_H
