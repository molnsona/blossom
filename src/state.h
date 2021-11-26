#ifndef STATE_H
#define STATE_H

//#define NO_CUDA
#include <Magnum/Magnum.h>

#include <memory>
#include <string>
#include <vector>

#include "data_model.h"
#include "graph_layout.h"
#include "keyboard_data.h"
#include "kmeans_landmark.h"
#include "knn_edges.h"
#include "landmark_model.h"
#include "mouse_data.h"
#include "scaled_data.h"
#include "scatter_model.h"
#include "training_config.h"
#include "trans_data.h"
#include "ui_data.h"

#include <Magnum/Magnum.h>

#include <vector>

using namespace Magnum;
using namespace Math::Literals;

struct State
{
    MouseData mouse;
    KeyboardData keyboard;

    DataModel data;
    RawDataStats stats;
    TransData trans;
    ScaledData scaled;
    LandmarkModel landmarks;

    GraphLayoutData layout_data;
    KMeansData kmeans_data;
    TrainingConfig training_conf;
    KnnEdgesData knn_data;

    ScatterModel scatter;

    State();

    void update(float time, UiData &ui);
};

#endif // #ifndef STATE_H
