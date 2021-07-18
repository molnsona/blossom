
#ifndef GRAPH_MODEL_H
#define GRAPH_MODEL_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <vector>

struct GraphModel
{
    GraphModel();

    static constexpr size_t side() { return 7; }

    std::vector<Magnum::Vector2> vertices;
    std::vector<float> edge_lengths;
    std::vector<std::pair<size_t, size_t>> edges; // constraint: first<second
};

#endif
