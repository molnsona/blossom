#ifndef STATE_H
#define STATE_H

#include <Magnum/Math/Vector.h>

#include <vector>

#include "imgui_config.h"
#include "pickable.hpp"

using namespace Magnum;
using namespace Math::Literals;

class State 
{
public:
    // Vector3 camera_trans{0.0f, 0.0f, 0.0f};

    // Vector2 zoom_depth{PLOT_WIDTH - 40, PLOT_HEIGHT - 40};

    // Vector2i mouse_press_pos;
    // Vector2i mouse_prev_pos;
    Vector2 mouse_pos;
    bool mouse_pressed{false};

    int cell_cnt{10000};
    int mean{0};
    int std_dev{300};

    std::vector<unsigned char> pixels = std::vector<unsigned char>(BYTES_PER_PIXEL * PLOT_WIDTH * PLOT_HEIGHT, DEFAULT_WHITE);
    std::vector<Vector2> vtx_pos;
    std::vector<Vector2i> edges;
    std::vector<int> lengths;

    bool vtx_selected{false};
   // bool mouse_released{false};
    UnsignedInt vtx_ind;
    // std::vector<PickableObject*> _vertices;

    std::size_t time = 0;
    std::size_t timeout = 1000;
    int expected_len = 100;
};

#endif // #ifndef STATE_H
