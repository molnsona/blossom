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

//#include <glm/glm.hpp>

//#include <tuple>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <iostream>

#include "wrapper_glfw.h"

// Defines several possible options for View movement. Used as abstraction to stay away from window-system specific input methods
enum View_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

// Default View values
const float SPEED       =  2.5f;
const float SENSITIVITY =  0.1f;
const float ZOOM        =  0.009f;//45.0f;


// An abstract View class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class View
{
public:
    // View Attributes
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;

    // View options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    // Framebuffer size
    int width;
    int height;

    // constructor with vectors
    View(int width = 800, int height = 600,
    glm::vec3 position = glm::vec3(0.0f, 0.0f,  1.0f), 
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f)) : 
        Front(glm::vec3(0.0f, 0.0f, -1.0f)), 
        MovementSpeed(SPEED), 
        MouseSensitivity(SENSITIVITY), 
        Zoom(ZOOM)   
    {
        Position = position;
        WorldUp = up;
        updateViewVectors();
    }
    // constructor with scalar values
    View(float posX, float posY, float posZ, float upX, float upY, float upZ,
    int width = 800, int height = 600) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
    {
        Position = glm::vec3(posX, posY, posZ);
        WorldUp = glm::vec3(upX, upY, upZ);
        updateViewVectors();
    }

    void update(float dt, const CallbackValues &callbacks)
    {
        width = callbacks.fb_width;
        height = callbacks.fb_height;

        ProcessKeyboard(callbacks.key, callbacks.key_action, dt);

        ProcessMouseScroll(callbacks.yoffset, dt);

        updateViewVectors();
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix() const
    {
        return glm::lookAt(Position, Position + Front, Up);
    }

    glm::mat4 GetProjMatrix() const
    {
        float half_w = width / 2.0f;
        float half_h = height / 2.0f;
        return glm::ortho(-half_w*Zoom, half_w*Zoom,-half_h*Zoom, half_h*Zoom, 0.1f, 100.0f);
        //return glm::perspective(glm::radians(Zoom), (float)width / (float)height, 0.1f, 100.0f);       
    }

    // // processes input received from any keyboard-like input system. Accepts input parameter in the form of View defined ENUM (to abstract it from windowing systems)
    // void ProcessKeyboard(View_Movement direction, float deltaTime)
    void ProcessKeyboard(int key, int action, float deltaTime)
    {
        float velocity = MovementSpeed * deltaTime;
        if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT))
            Position += Up * velocity;
        if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT))
            Position -= Up * velocity;
        if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT))
            Position -= glm::normalize(glm::cross(Front, Up)) * velocity;
        if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT))
            Position += glm::normalize(glm::cross(Front, Up)) * velocity;
    }

    // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void ProcessMouseMovement(float xoffset, float yoffset)
    {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        // update Front, Right and Up Vectors using the updated Euler angles
        updateViewVectors();
    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void ProcessMouseScroll(float yoffset, float deltaTime)
    {
        float cameraSpeed = static_cast<float>(2.5 * deltaTime);

        if(yoffset > 0)
            Zoom -= 0.0005;
        else if(yoffset < 0)
            Zoom += 0.0005;
            
        // float velocity = MovementSpeed * deltaTime * (Zoom / 20);

        // if(yoffset > 0)
        //     Position += velocity * Front;
        // else if(yoffset < 0)
        //     Position -= velocity * Front;
        
        // if(Position.z < 0.2) Position.z = 0.2;
        
        // //Zoom -= (float)yoffset * 2;

        // Zoom -= (float)yoffset * 2 * (Zoom / 20);
        // if (Zoom < 1.0f)
        //     Zoom = 1.0f;
        // if (Zoom > 45.0f)
        //     Zoom = 45.0f;         
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
        glm::vec3 res = glm::unProject(
            glm::vec3(mouse.x, height - mouse.y, 0.1f), 
            GetViewMatrix(), 
            GetProjMatrix(), 
            glm::vec4(0.0f, 0.0f, width, height));

        return glm::vec2(res.x, res.y);
    }

    /**
     * @brief Converts point to screen coordinates([0,0] in the middle of the screen).
     * 
     * @param point Input point in original not projected coordinates.
     * @return glm::vec2 Point in screen coordinates.
     */
    glm::vec2 screen_coords(glm::vec2 point) const
    {
        glm::vec3 res = glm::project(glm::vec3(point, 0.1f), GetViewMatrix(), GetProjMatrix(), glm::vec4(0.0f, 0.0f, width, height));
        return screen_point_coords(glm::vec2(res.x, res.y));
    }
    
private:
    // calculates the front vector from the View's (updated) Euler Angles
    void updateViewVectors()
    {
        // also re-calculate the Right and Up vector
        Right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up    = glm::normalize(glm::cross(Right, Front));
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

// /**
//  * @brief A small utility class that manages the viewport coordinates, together
//  * with the virtual "camera" position and zoom.
//  */
// struct View
// {
//     using Vector2 = glm::vec2;
//     using Vector2i = glm::ivec2;
//     using Vector3 = glm::vec3;
//     using Matrix3 = glm::mat3;

//     /** Size of the framebuffer (in screen coordinates). */
//     Vector2i fb_size;

//     /** Current middle point of the view (in model coordinates). */
//     Vector2 mid;
//     /** Intended middle point of the view (camera will move there). */
//     Vector2 mid_target;
//     /** Negative-log2-vertical-size of the on-screen part of the model space
//      * (aka zoom). */
//     float view_logv;
//     /** Intended zoom. */
//     float view_logv_target;

//     View()
//       : fb_size(1, 1)
//       , mid(4, 4)
//       , mid_target(4, 4)
//       , view_logv(-4)
//       , view_logv_target(-4)
//     {}

//     /** Utility to convert logv to actual size */
//     inline float zoom_scale(float logv) const { return pow(2, -logv); }

//     /** Return the vertical size of the viewed part in model space. */
//     inline float zoom_scale() const { return zoom_scale(view_logv); }

//     /**
//      * @brief Move the current midpoint and zoom a bit closer to the target
//      * midpoint and zoom.
//      *
//      * @param dt Time difference
//      */
//     void update(float dt)
//     {
//         const float ir = pow(0.005, dt);
//         const float r = 1 - ir;

//         view_logv = ir * view_logv + r * view_logv_target;
//         mid = ir * mid + r * mid_target;
//     }

//     /**
//      * @brief Compute and return the view size in model coordinates
//      *
//      * @return Vector2 with the size.
//      */
//     inline Vector2 view_size() const
//     {
//         const float v = zoom_scale();
//         const float h = fb_size.x * v / fb_size.y;
//         return Vector2(h, v);
//     }

//     /**
//      * @brief Compute the view rectangle coordinates
//      *
//      * @return Tuple of two Vector2, with the lower-left view corner and size of
//      * the view.
//      */
//     inline std::tuple<Vector2, Vector2> frame() const
//     {
//         auto s = view_size();
//         return { Vector2((mid - s).x / 2, (mid - s).y / 2), s };
//     }

//     /**
//      * @brief Compute model coordinates from screen coordinates.
//      *
//      * The "zero" point is thought to be in the middle of the screen.
//      *
//      * @param screen Screen coordinates.
//      * @return Model coordinates.
//      */
//     inline Vector2 model_coords(Vector2i screen) const
//     {
//         return mid + view_size() *
//                        (Vector2(screen) / Vector2(fb_size) - Vector2(0.5, 0.5));
//     }

//     /**
//      * @brief Computes screen coordinates from model coordinates.
//      *
//      * Inverse to model_coords().
//      *
//      * @param model Model coordinates.
//      * @return Screen coordinates.
//      */
//     inline Vector2 screen_coords(Vector2 model) const
//     {
//         return ((model - mid) / view_size() + Vector2(0.5, 0.5)) *
//                Vector2(fb_size);
//     }

//     /**
//      * @brief Computes screen coordinates of the mouse cursor.
//      *
//      * @param mouse Mouse cursor coordinates ([0, 0] is in the bottom left
//      * corner.)
//      * @return Screen coordinates of the mouse cursor.
//      */
//     inline Vector2i screen_mouse_coords(Vector2i mouse) const
//     {
//         return Vector2i(mouse.x, fb_size.y - mouse.y);
//     }

//     /**
//      * @brief Computes model coordinates of the mouse cursor.
//      *
//      * @param mouse Mouse cursor coordinates ([0, 0] is in the bottom left
//      * corner.)
//      * @return Model coordinates of the mouse cursor.
//      */
//     inline Vector2 model_mouse_coords(Vector2i mouse) const
//     {
//         return model_coords(screen_mouse_coords(mouse));
//     }

//     /**
//      * @brief Compute the projection matrix for drawing into the "view" space.
//      *
//      * 1 unit in the view space should be roughly equal to 1 framebuffer pixel,
//      * independently of camera position and zoom.
//      */
//     inline glm::mat3 screen_projection_matrix() const
//     {
//         return glm::mat3(Vector3(2.0f / fb_size.x, 0, 0),
//                                Vector3(0, 2.0f / fb_size.y, 0),
//                                Vector3(-1, -1, 1));
//     }

//     /**
//      * @brief Compute the projection matrix for drawing into the "model"
//      * coordinates.
//      *
//      * This view is transformed along with camera position an zoom.
//      */
//     inline glm::mat3 projection_matrix() const
//     {
//         auto isize = Vector2(1.0 / view_size().x, 1.0 / view_size().y);

//         return glm::mat3(
//           Vector3(2 * isize.x, 0, 0),
//           Vector3(0, 2 * isize.y, 0),
//           Vector3(-2 * mid.x * isize.x, -2 * mid.y * isize.y, 1));
//     }

//     /**
//      * @brief Cause a zoom in response to user action.
//      *
//      * Call this to handle mousewheel scrolling events.
//      */
//     void zoom(float delta, Vector2i mouse)
//     {
//         view_logv_target += delta;
//         if (view_logv_target > 15)
//             view_logv_target = 15;
//         if (view_logv_target < -10)
//             view_logv_target = -10;

//         auto zoom_around = model_mouse_coords(mouse);
//         mid_target = zoom_around + zoom_scale(view_logv_target) *
//                                      (mid - zoom_around) / zoom_scale();
//     }

//     /**
//      * @brief Cause the camera to look at the specified point.
//      *
//      * @param tgt This point will eventually get to the middle of the screen.
//      */
//     void lookat(Vector2 tgt) { mid_target = tgt; }

//     /** Reset the framebuffer size to the specified value */
//     void set_fb_size(Vector2i s) { fb_size = s; }

//     /** Variant of lookat() that accepts "screen" coordinates. */
//     void lookat_screen(Vector2i mouse) { lookat(model_mouse_coords(mouse)); }
// };

#endif
