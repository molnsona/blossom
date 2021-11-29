
#ifndef TSNE_LAYOUT_H
#define TSNE_LAYOUT_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

#include <vector>

#include "landmark_model.h"
#include "mouse_data.h"

struct TSNELayoutData
{
    std::vector<float> pji;
    std::vector<size_t> heap;
    std::vector<Vector2> updates;
};

void
tsne_layout_step(TSNELayoutData &data,
                 const MouseData &mouse,
                 LandmarkModel &lm,
                 float time);
#endif
