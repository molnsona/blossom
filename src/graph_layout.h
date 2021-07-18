
#ifndef LAYOUT_H
#define LAYOUT_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

#include <vector>

struct GraphLayoutData
{
    std::vector<Magnum::Vector2> velocities;
    std::vector<Magnum::Vector2> forces; // kept allocated for efficiency
};

void
graph_layout_step(GraphLayoutData &data,
                  std::vector<Magnum::Vector2> &vertices,
                  const std::vector<std::pair<size_t, size_t>> &edges,
                  const std::vector<float> &lengths,
                  float time);

#endif
