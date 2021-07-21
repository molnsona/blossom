#ifndef STATE_H
#define STATE_H

//#define NO_CUDA
#include <Magnum/Magnum.h>

#include <string>
#include <vector>

#include "imgui_config.h"
#include "pickable.hpp"

#include "data_model.h"
#include "fcs_parser.h"
#include "graph_layout.h"
#include "landmark_model.h"
#include "mouse_data.h"
#include "scatter_model.h"
#include "embedsom_cuda.h"

#include <Magnum/Magnum.h>

#include <vector>


using namespace Magnum;
using namespace Math::Literals;

struct State
{
    int cell_cnt{ 10000 };
    int mean{ 0 };
    int std_dev{ 300 };
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

    bool parse{ false };
    std::string file_path;
    FCSParser fcs_parser;

    DataModel data;
    LandmarkModel landmarks;
    ScatterModel scatter;

    GraphLayoutData layout_data;
#ifndef NO_CUDA
    EsomCuda esom_cuda;
#endif;

    void update(float time);
};

#endif // #ifndef STATE_H
