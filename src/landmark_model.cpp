
#include "landmark_model.h"

#include <random>

LandmarkModel::LandmarkModel()
  : d(2 /*TODO take from data model*/)
{
    auto n = side();

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
LandmarkModel::update(const DataModel &data)
{
    d = data.d;
    auto n = side();

    hidim_vertices.resize(n * n * d);

    std::size_t index = 0;
    for (size_t i = 100; i < 500; i += 400/n) {
        for (int j = -150; j < 150; j += 300 / n) {
            float y = i;
            float z = j;

            // TODO compute this dynamically from data model
            hidim_vertices[d * index + 0] = 0;
            hidim_vertices[d * index + 1] = y;
            hidim_vertices[d * index + 2] = z;
            ++index;
        }
    }
}
