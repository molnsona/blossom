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

#ifndef VIEW_H
#define VIEW_H

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <iostream>
#include <vector>

#include "wrapper_glfw.h"

/**
 * @brief A small utility class that manages the viewport coordinates, together
 * with the virtual "camera" position and zoom.
 */
class View
{
public:
    // Default View values
    const float movement_speed = 0.2f;
    const float smooth_speed = 0.005f;
    const float zoom_depth = 0.009f;

    // View vectors
    glm::vec3 target_pos;
    glm::vec3 current_pos;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 world_up;

    float target_zoom;
    float current_zoom;

    // Framebuffer size
    int width;
    int height;

    View(int width = 800,
         int height = 600,
         glm::vec3 position = glm::vec3(0.0f, 0.0f, 1.0f),
         glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f))
      : front(glm::vec3(0.0f, 0.0f, -1.0f))
      , target_zoom(zoom_depth)
      , current_zoom(zoom_depth)
    {
        target_pos = position;
        current_pos = position;
        world_up = up;
        update_view_vectors();
    }

    /**
     * @brief Move the current position and zoom a bit closer to the target
     * position and zoom.
     *
     * @param dt Time difference.
     * @param callbacks Values collected by event callbacks.
     */
    void update(float dt, const CallbackValues &callbacks)
    {
        width = callbacks.fb_width;
        height = callbacks.fb_height;

        process_keyboard(callbacks.key, callbacks.key_action);

        process_mouse_scroll(callbacks.yoffset,
                             glm::vec2(callbacks.xpos, callbacks.ypos));

        float power = pow(smooth_speed, dt);
        float r = 1 - power;
        current_pos = power * current_pos + r * target_pos;
        current_zoom = power * current_zoom + r * target_zoom;

        update_view_vectors();
    }

    /**
     * @brief Compute view matrix for drawing into the "view" space.
     *
     * @return glm::mat4 View matrix
     */
    glm::mat4 get_view_matrix() const
    {
        return glm::lookAt(current_pos, current_pos + front, up);
    }

    /**
     * @brief Compute projection matrix for orthographic projection.
     *
     * @return glm::mat4 Projection matrix
     */
    glm::mat4 get_proj_matrix() const
    {
        float half_w = width / 2.0f;
        float half_h = height / 2.0f;
        return glm::ortho(-half_w * current_zoom,
                          half_w * current_zoom,
                          -half_h * current_zoom,
                          half_h * current_zoom,
                          0.1f,
                          100.0f);
    }

    /**
     * @brief Convert mouse coordinates ([0,0] in the upper left corner),
     * to screen coordinates ([0,0] in the middle of the screen).
     *
     * @param mouse Mouse coordinates ([0,0] in the upper left corner)
     * @return glm::vec2 Screen coordinates ([0,0] in the middle of the screen)
     */
    glm::vec2 screen_mouse_coords(glm::vec2 mouse) const
    {
        return glm::vec2(mouse.x - width / 2.0f, height / 2.0f - mouse.y);
    }

    /**
     * @brief Convert mouse coordinates ([0,0] in the upper left corner),
     * to model coordinates ([0,0] in the middle of the screen).
     *
     * @param mouse Mouse coordinates ([0,0] in the upper left corner)
     * @return glm::vec2 Model coordinates ([0,0] in the middle of the screen)
     */
    glm::vec2 model_mouse_coords(glm::vec2 mouse) const
    {
        glm::vec3 res =
          glm::unProject(glm::vec3(mouse.x, height - mouse.y, 0.1f),
                         get_view_matrix(),
                         get_proj_matrix(),
                         glm::vec4(0.0f, 0.0f, width, height));

        return glm::vec2(res.x, res.y);
    }

    /**
     * @brief Converts point to screen coordinates([0,0] in the middle of the
     * screen).
     *
     * @param point Input point in original not projected coordinates.
     * @return glm::vec2 Point in screen coordinates.
     */
    glm::vec2 screen_coords(glm::vec2 point) const
    {
        glm::vec3 res = glm::project(glm::vec3(point, 0.1f),
                                     get_view_matrix(),
                                     get_proj_matrix(),
                                     glm::vec4(0.0f, 0.0f, width, height));
        return screen_point_coords(glm::vec2(res.x, res.y));
    }

private:
    /**
     * @brief Re-calculates the right and up vector from the View's updated
     * vectors.
     *
     */
    void update_view_vectors()
    {
        right = glm::normalize(glm::cross(
          front, world_up)); // normalize the vectors, because their length gets
                             // closer to 0 the more you look up or down which
                             // results in slower movement.
        up = glm::normalize(glm::cross(right, front));
    }

    /**
     * @brief Process keyboard input and move the "camera" accordingly.
     *
     * @param key Name of the used key.
     * @param action Pressed, released or hold key.
     */
    void process_keyboard(int key, int action)
    {
        float half_h = height / 2.0f;
        float velocity = half_h * target_zoom * movement_speed;
        if (key == GLFW_KEY_W &&
            (action == GLFW_PRESS || action == GLFW_REPEAT))
            target_pos.y += velocity;
        if (key == GLFW_KEY_S &&
            (action == GLFW_PRESS || action == GLFW_REPEAT))
            target_pos.y -= velocity;
        if (key == GLFW_KEY_A &&
            (action == GLFW_PRESS || action == GLFW_REPEAT))
            target_pos.x -= velocity;
        if (key == GLFW_KEY_D &&
            (action == GLFW_PRESS || action == GLFW_REPEAT))
            target_pos.x += velocity;
    }

    /**
     * @brief Process mouse scroll and adjust zoom accordingly.
     *
     * @param yoffset Direction of the scroll (-1, 0, 1).
     * @param mouse Mouse coordinates ([0,0] in the upper left corner).
     */
    void process_mouse_scroll(float yoffset, glm::vec2 mouse)
    {
        if (yoffset > -0.0001 && yoffset < 0.0001)
            return;

        float velocity = -1 * yoffset / 1500.0f * (target_zoom * 100);
        auto zoom_around = model_mouse_coords(mouse);

        target_zoom += velocity;
        target_pos =
          glm::vec3(zoom_around + powf(2.0f, target_zoom * 400) *
                                    (glm::vec2(current_pos) - zoom_around) /
                                    powf(2.0f, current_zoom * 400),
                    target_pos.z);
    }

    /**
     * @brief Convert point coordinates ([0,0] in the upper left corner),
     * to screen coordinates ([0,0] in the middle of the screen).
     *
     * @param point Point coordinates ([0,0] in the upper left corner)
     * @return glm::vec2 Screen coordinates ([0,0] in the middle of the screen)
     */
    glm::vec2 screen_point_coords(glm::vec2 point) const
    {
        return glm::vec2(point.x - width / 2.0f, point.y - height / 2.0f);
    }
};

#endif
