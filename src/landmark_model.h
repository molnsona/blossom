
#ifndef GRAPH_MODEL_H
#define GRAPH_MODEL_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <vector>

#include "data_model.h"

struct LandmarkModel
{
    LandmarkModel();

    static constexpr size_t side() { return 7; }

    void update(const DataModel &data);

    std::vector<float> hidim_vertices;
    size_t d;

    std::vector<Magnum::Vector2> lodim_vertices;

    std::vector<float> edge_lengths;
    std::vector<std::pair<size_t, size_t>> edges; // constraint: first<second
};

#endif
