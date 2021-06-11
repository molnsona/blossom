#ifndef STATE_H
#define STATE_H

#include <Magnum/Math/Vector.h>

#include <vector>

#include "../ui/imgui_config.h"
#include "../pickable.hpp"

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

    std::vector<unsigned char> pixels = std::vector<unsigned char>(BYTES_PER_PIXEL * PLOT_WIDTH * PLOT_HEIGHT, DEFAULT_WHITE);
    std::vector<Vector2i> vtx_pos;
    std::vector<Vector2i> edges;

    std::vector<PickableObject*> _vertices;
};

#endif // #ifndef STATE_H