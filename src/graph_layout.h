
#ifndef LAYOUT_H
#define LAYOUT_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

#include <vector>

#include "landmark_model.h"
#include "mouse_data.h"

struct GraphLayoutData
{
    std::vector<Magnum::Vector2> velocities;
    std::vector<Magnum::Vector2> forces; // kept allocated for efficiency
};

void
graph_layout_step(GraphLayoutData &data,
                  const MouseData &mouse,
                  LandmarkModel &lm,
                  float time);

#endif
