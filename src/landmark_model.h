
#ifndef GRAPH_MODEL_H
#define GRAPH_MODEL_H

#include "view.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <vector>

#include "dirty.h"

struct LandmarkModel : public Dirt
{
    size_t d;
    std::vector<float> hidim_vertices;
    std::vector<Magnum::Vector2> lodim_vertices;

    std::vector<float> edge_lengths;
    std::vector<std::pair<size_t, size_t>> edges; // constraint: first<second

    LandmarkModel();
    void update_dim(size_t dim);
    void init_grid(size_t side);

    void press(const size_t &ind,
               const Magnum::Vector2i &mouse_pos,
               View &view);
    void move(const size_t &ind, const Magnum::Vector2i &mouse_pos, View &view);
    void duplicate(const size_t &ind);
    void remove(const size_t &ind);

    size_t n_landmarks() const { return lodim_vertices.size(); }
};

#endif
