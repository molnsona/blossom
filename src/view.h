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

#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>
#include <tuple>

/**
 * @brief A small utility class that manages the viewport coordinates, together
 * with the virtual "camera" position and zoom.
 */
struct View
{
    using Vector2 = Magnum::Vector2;
    using Vector2i = Magnum::Vector2i;
    using Vector3 = Magnum::Vector3;
    using Matrix3 = Magnum::Matrix3;

    /** Size of the framebuffer (in screen coordinates). */
    Vector2i fb_size;

    /** Current middle point of the view (in model coordinates). */
    Vector2 mid;
    /** Intended middle point of the view (camera will move there). */
    Vector2 mid_target;
    /** Negative-log2-vertical-size of the on-screen part of the model space (aka zoom). */
    float view_logv;
    /** Intended zoom. */
    float view_logv_target;

    View()
      : fb_size(1, 1)
      , mid(4, 4)
      , mid_target(4, 4)
      , view_logv(-4)
      , view_logv_target(-4)
    {}

    /** Utility to convert logv to actual size */
    inline float zoom_scale(float logv) const { return pow(2, -logv); }

    /** Return the vertical size of the viewed part in model space. */
    inline float zoom_scale() const { return zoom_scale(view_logv); }

    /**
     * @brief Move the current midpoint and zoom a bit closer to the target midpoint and zoom.
     *
     * @param dt Time difference
     */
    void update(float dt)
    {
        const float ir = pow(0.005, dt);
        const float r = 1 - ir;

        view_logv = ir * view_logv + r * view_logv_target;
        mid = ir * mid + r * mid_target;
    }

    /**
     * @brief Compute and return the view size in model coordinates
     *
     * @return Vector2 with the size.
     */
    inline Vector2 view_size() const
    {
        const float v = zoom_scale();
        const float h = fb_size.x() * v / fb_size.y();
        return Vector2(h, v);
    }

    /**
     * @brief Compute the view rectangle coordinates
     *
     * @return Tuple of two Vector2, with the lower-left view corner and size of the view.
     */
    inline std::tuple<Vector2, Vector2> frame() const
    {
        auto s = view_size();
        return { mid - s / 2, s };
    }

    /**
     * @brief Compute model coordinates from screen coordinates.
     *
     * The "zero" point is thought to be in the middle of the screen.
     *
     * @param screen Screen coordinates.
     * @return Model coordinates.
     */
    inline Vector2 model_coords(Vector2i screen) const
    {
        return mid + view_size() *
                       (Vector2(screen) / Vector2(fb_size) - Vector2(0.5, 0.5));
    }

    /**
     * @brief Computes screen coordinates from model coordinates.
     *
     * Inverse to model_coords().
     *
     * @param model Model coordinates.
     * @return Screen coordinates.
     */
    inline Vector2 screen_coords(Vector2 model) const
    {
        return ((model - mid) / view_size() + Vector2(0.5, 0.5)) *
               Vector2(fb_size);
    }

    /**
     * @brief Computes screen coordinates of the mouse cursor.
     *
     * @param mouse Mouse cursor coordinates ([0, 0] is in the bottom left
     * corner.)
     * @return Screen coordinates of the mouse cursor.
     */
    inline Vector2i screen_mouse_coords(Vector2i mouse) const
    {
        return Vector2i(mouse.x(), fb_size.y() - mouse.y());
    }

    /**
     * @brief Computes model coordinates of the mouse cursor.
     *
     * @param mouse Mouse cursor coordinates ([0, 0] is in the bottom left
     * corner.)
     * @return Model coordinates of the mouse cursor.
     */
    inline Vector2 model_mouse_coords(Vector2i mouse) const
    {
        return model_coords(screen_mouse_coords(mouse));
    }

    /**
     * @brief Compute the projection matrix for drawing into the "view" space.
     *
     * 1 unit in the view space should be roughly equal to 1 framebuffer pixel,
     * independently of camera position and zoom.
     */
    inline Magnum::Matrix3 screen_projection_matrix() const
    {
        return Magnum::Matrix3(Vector3(2.0f / fb_size.x(), 0, 0),
                               Vector3(0, 2.0f / fb_size.y(), 0),
                               Vector3(-1, -1, 1));
    }

    /**
     * @brief Compute the projection matrix for drawing into the "model" coordinates.
     *
     * This view is transformed along with camera position an zoom.
     */
    inline Magnum::Matrix3 projection_matrix() const
    {
        auto isize = 1.0 / view_size();

        return Magnum::Matrix3(
          Vector3(2 * isize.x(), 0, 0),
          Vector3(0, 2 * isize.y(), 0),
          Vector3(-2 * mid.x() * isize.x(), -2 * mid.y() * isize.y(), 1));
    }

    /**
     * @brief Cause a zoom in response to user action.
     *
     * Call this to handle mousewheel scrolling events.
     */
    void zoom(float delta, Vector2i mouse)
    {
        view_logv_target += delta;
        if (view_logv_target > 15)
            view_logv_target = 15;
        if (view_logv_target < -10)
            view_logv_target = -10;

        auto zoom_around = model_mouse_coords(mouse);
        mid_target = zoom_around + zoom_scale(view_logv_target) *
                                     (mid - zoom_around) / zoom_scale();
    }

    /**
     * @brief Cause the camera to look at the specified point.
     *
     * @param tgt This point will eventually get to the middle of the screen.
     */
    void lookat(Vector2 tgt) { mid_target = tgt; }

    /** Reset the framebuffer size to the specified value */
    void set_fb_size(Vector2i s) {
        fb_size = s;
    }

    /** Variant of lookat() that accepts "screen" coordinates. */
    void lookat_screen(Vector2i mouse) { lookat(model_mouse_coords(mouse)); }
};

#endif
