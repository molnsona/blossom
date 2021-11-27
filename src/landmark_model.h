
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

    void press(size_t ind, const Magnum::Vector2 &mouse_pos);
    void move(size_t ind, const Magnum::Vector2 &mouse_pos);
    void duplicate(size_t ind);
    void add(const Magnum::Vector2 &mouse_pos);
    void remove(size_t ind);

    size_t closest_landmark(const Magnum::Vector2 &mouse_pos) const;

    size_t n_landmarks() const { return lodim_vertices.size(); }
};

#endif
