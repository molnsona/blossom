#ifndef STATE_H
#define STATE_H

#include <Magnum/Math/Vector.h>

#include "../ui/imgui_config.h"

using namespace Magnum;
using namespace Math::Literals;

class State 
{
public:
    Vector3 camera_trans{0.0f, 0.0f, 0.0f};

    Vector2 zoom_depth{PLOT_WIDTH - 40, PLOT_HEIGHT - 40};

    Vector2i mouse_press_pos;
    Vector2d mouse_delta;

    int cell_cnt = 10000;
    int mean = 0;
    int std_dev = 300;
};

#endif // #ifndef STATE_H