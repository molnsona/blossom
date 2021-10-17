
#ifndef GRAPH_MODEL_H
#define GRAPH_MODEL_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <vector>

struct LandmarkModel
{
    size_t d;
    std::vector<float> hidim_vertices;
    std::vector<Magnum::Vector2> lodim_vertices;

    std::vector<float> edge_lengths;
    std::vector<std::pair<size_t, size_t>> edges; // constraint: first<second

    LandmarkModel();
    void update_dim(size_t dim);
    void init_grid(size_t side);

    size_t n_landmarks() const { return lodim_vertices.size(); }
};

#endif
