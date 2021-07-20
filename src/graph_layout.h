
#ifndef LAYOUT_H
#define LAYOUT_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

#include <vector>

#include "mouse_data.h"

struct GraphLayoutData
{
    std::vector<Magnum::Vector2> velocities;
    std::vector<Magnum::Vector2> forces; // kept allocated for efficiency
};

void
graph_layout_step(GraphLayoutData &data,
                  const MouseData &mouse,
                  std::vector<Magnum::Vector2> &vertices,
                  const std::vector<std::pair<size_t, size_t>> &edges,
                  const std::vector<float> &lengths,
                  float time);

#endif
