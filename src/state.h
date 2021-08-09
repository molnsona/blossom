#ifndef STATE_H
#define STATE_H

//#define NO_CUDA
#include <Magnum/Magnum.h>

#include <memory>
#include <string>
#include <vector>

#if 0
#include "imgui_config.h"
#include "pickable.hpp"
#endif

#include "data_model.h"
#include "embedsom_cuda.h"
#include "fcs_parser.h"
#include "graph_layout.h"
#include "landmark_model.h"
#include "mouse_data.h"
#include "scatter_model.h"
#include "trans_data.h"
#include "ui_data.h"

#include <Magnum/Magnum.h>

#include <vector>

using namespace Magnum;
using namespace Math::Literals;

struct State
{
#if 0
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
#endif
    MouseData mouse;

    UiData ui;

    DataModel data;
    TransData trans;
    LandmarkModel landmarks;
    ScatterModel scatter;

    GraphLayoutData layout_data;
#ifndef NO_CUDA
    EsomCuda esom_cuda;
#endif;

    State();

    void update(float time);
};

#endif // #ifndef STATE_H
