#ifndef STATE_H
#define STATE_H

#include <Magnum/Magnum.h>

#include <vector>

#include "imgui_config.h"
#include "pickable.hpp"

#include "data_model.h"
#include "graph_layout.h"
#include "landmark_model.h"
#include "mouse_data.h"
#include "scatter_model.h"

using namespace Magnum;
using namespace Math::Literals;

struct State
{
    // Vector2i _mouse_press_pos;
    // Vector2i _mouse_prev_pos;

    // Vector2 mouse_pos;
    // bool mouse_pressed{ false };

    int cell_cnt{ 10000 };
    int mean{ 0 };
    int std_dev{ 300 };

    std::vector<unsigned char> pixels =
      std::vector<unsigned char>(BYTES_PER_PIXEL * PLOT_WIDTH * PLOT_HEIGHT,
                                 DEFAULT_WHITE);
    std::vector<Vector2> vtx_pos;
    std::vector<Vector2i> edges;
    std::vector<float> lengths;

    bool vtx_selected{ false };
    UnsignedInt vtx_ind;

    std::size_t time = 0;
    std::size_t timeout = 1000;
    int expected_len = 100;

    MouseData mouse;

    DataModel data;
    LandmarkModel landmarks;
    ScatterModel scatter;

    GraphLayoutData layout_data;

    void update(float time);
};

#endif // #ifndef STATE_H
