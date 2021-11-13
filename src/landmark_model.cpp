
#include "landmark_model.h"

#include <random>

LandmarkModel::LandmarkModel()
  : d(0)
{
    init_grid(0);
}

void
LandmarkModel::init_grid(size_t n)
{
    if (!d) {
        lodim_vertices.clear();
        return;
    }

    lodim_vertices.resize(n * n);
    hidim_vertices.resize(n * n * d);

    edge_lengths.reserve(2 * n * (n - 1));
    edges.reserve(edge_lengths.size());

    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.5, 1.5);

    for (size_t i = 0; i < n * n; ++i) {
        auto x = i % n;
        auto y = i / n;
        lodim_vertices[i] = Magnum::Vector2(x, y);

        // TODO compute this dynamically from data model
        hidim_vertices[d * i + 0] = x / float(n);
        hidim_vertices[d * i + 1] = y / float(n);

        if (x + 1 < n) {
            edges.emplace_back(i, x + 1 + y * n);
            edge_lengths.push_back(dist(gen) + 3 * x / float(n));
        }
        if (y + 1 < n) {
            edges.emplace_back(i, x + (y + 1) * n);
            edge_lengths.push_back(dist(gen) + 3 * y / float(n));
        }
    }
}

void
LandmarkModel::update_dim(size_t dim)
{
    if (dim == d)
        return;
    d = dim;
    init_grid(5);
}

void
LandmarkModel::press(const std::size_t &ind,
                     const Magnum::Vector2i &mouse_pos,
                     View &view)
{
    lodim_vertices[ind] = view.model_mouse_coords(mouse_pos);
}

void
LandmarkModel::move(const std::size_t &ind,
                    const Magnum::Vector2i &mouse_pos,
                    View &view)
{
    lodim_vertices[ind] = view.model_mouse_coords(mouse_pos);
}

void
LandmarkModel::duplicate(const std::size_t &ind)
{
    // Add new line to hidim
    std::size_t line_idx = d * ind;
    for (std::size_t i = 0; i < d; ++i) {
        hidim_vertices.emplace_back(hidim_vertices[line_idx + i]);
    }

    // Add new vertex to lodim
    lodim_vertices.emplace_back(
      Magnum::Vector2(lodim_vertices[ind].x() + 0.3, lodim_vertices[ind].y()));
    std::size_t new_vert_ind = lodim_vertices.size() - 1;
#if 0
    // Find edges.
    std::vector<std::size_t> edge_idxs;
    for(std::size_t i = 0; i < edges.size(); ++i) {
        if(edges[i].first == ind || edges[i].second == ind) {
            edge_idxs.emplace_back(i);         
        }
    }    

    // Add new edges and edge lengths
    for(std::size_t i = 0; i < edge_idxs.size(); ++i) {
        std::size_t edge_idx = edge_idxs[i];
        auto edge = edges[edge_idx];
        if(edge.first == ind)
            edges.emplace_back(std::make_pair(new_vert_ind, edge.second));
        else if(edge.second == ind)
            edges.emplace_back(std::make_pair(edge.first, new_vert_ind));
        
        edge_lengths.emplace_back(edge_lengths[edge_idx]);
    }
#endif
}

void
LandmarkModel::remove(const std::size_t &ind)
{
    lodim_vertices.erase(lodim_vertices.begin() + ind);
    std::size_t line_idx = d * ind;
    hidim_vertices.erase(hidim_vertices.begin() + line_idx,
                         hidim_vertices.begin() + line_idx + 4);

    // Remove edges.
    std::vector<std::size_t> edge_idxs;
    std::size_t edge_ind = 0;
    for (auto i = edges.begin(); i != edges.end();) {
        if (i->first == ind || i->second == ind) {
            i = edges.erase(i);
            edge_lengths.erase(edge_lengths.begin() + edge_ind);
            continue;
        }
        ++edge_ind;
        ++i;
    }

    // Update indices of vertices that are after the removing vertex
    // so the edges have proper vertex indices
    for (auto i = edges.begin(); i != edges.end(); ++i) {
        if (i->first >= ind) {
            --i->first;
        }
        if (i->second >= ind) {
            --i->second;
        }
    }
}
