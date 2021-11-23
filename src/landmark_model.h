
#ifndef GRAPH_MODEL_H
#define GRAPH_MODEL_H

#include "view.h"
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

    void press(const std::size_t &ind,
               const Magnum::Vector2i &mouse_pos,
               View &view);
    void move(const std::size_t &ind,
              const Magnum::Vector2i &mouse_pos,
              View &view);
    void duplicate(const std::size_t &ind);
    void remove(const std::size_t &ind);

    size_t n_landmarks() const { return lodim_vertices.size(); }
};

#endif
