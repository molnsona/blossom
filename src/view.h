
#ifndef VIEW_H
#define VIEW_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>
#include <tuple>

struct View
{
    using Vector2 = Magnum::Vector2;
    using Vector2i = Magnum::Vector2i;
    using Vector3 = Magnum::Vector3;
    using Matrix3 = Magnum::Matrix3;

    Vector2i fb_size;
    Vector2 mid, mid_target;
    float view_logv, view_logv_target;

    View()
      : fb_size(1, 1)
      , mid(0, 0)
      , mid_target(0, 0)
      , view_logv(0)
      , view_logv_target(0)
    {}

    void update(float dt)
    {
        const float ir = pow(0.005, dt);
        const float r = 1 - ir;

        view_logv = ir * view_logv + r * view_logv_target;
        mid = ir * mid + r * mid_target;
    }

    inline Vector2 view_size() const
    {
        const float v = pow(2, -view_logv);
        const float h = fb_size.x() * v / fb_size.y();
        return Vector2(h, v);
    }

    inline std::tuple<Vector2, Vector2> frame() const
    {
        auto s = view_size();
        return { mid - s / 2, s };
    }

    inline Vector2 model_coords(Vector2i screen) const
    {
        return mid + view_size() *
                       (Vector2(screen) / Vector2(fb_size) - Vector2(0.5, 0.5));
    }

    inline Vector2 screen_coords(Vector2 model) const
    {
        return ((model - mid) / view_size() + Vector2(0.5, 0.5)) *
               Vector2(fb_size);
    }

    inline Magnum::Matrix3 screen_projection_matrix() const
    {
        return Magnum::Matrix3(Vector3(2.0f / fb_size.x(), 0, 0),
                               Vector3(0, 2.0f / fb_size.y(), 0),
                               Vector3(-1, -1, 1));
    }

    inline Magnum::Matrix3 projection_matrix() const
    {
        auto isize = 1.0 / view_size();

        return Magnum::Matrix3(
          Vector3(2 * isize.x(), 0, 0),
          Vector3(0, 2 * isize.y(), 0),
          Vector3(-2 * mid.x() * isize.x(), -2 * mid.y() * isize.y(), 1));
    }

    // kinda event handlers

    void zoom(float delta)
    {
        /*TODO: this zooms around the "mid" point, should instead zoom around
         * mouse cursor */
        view_logv_target += delta;
        if (view_logv_target > 15)
            view_logv_target = 15;
        if (view_logv_target < -10)
            view_logv_target = -10;
    }

    void lookat(Vector2 tgt) { mid_target = tgt; }

    void set_fb_size(Vector2i);
};

#endif
