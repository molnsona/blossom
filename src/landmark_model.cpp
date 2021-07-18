
#include "landmark_model.h"

LandmarkModel::LandmarkModel()
{
    auto n = side();

    vertices.resize(n * n);
    edge_lengths.reserve(2 * n * (n - 1));
    edges.reserve(edge_lengths.size());

    for (size_t i = 0; i < n * n; ++i) {
        auto x = i % n;
        auto y = i / n;
        vertices[i] = Magnum::Vector2(x, y);
        if (x + 1 < n) {
            edges.emplace_back(i, x + 1 + y * n);
            edge_lengths.push_back(1);
        }
        if (y + 1 < n) {
            edges.emplace_back(i, x + (y + 1) * n);
            edge_lengths.push_back(1);
        }
    }
}
