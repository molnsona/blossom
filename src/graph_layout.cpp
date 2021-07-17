
#include "graph_layout.h"

// TODO: parametrize the constants

void
graph_layout_step(GraphLayoutData &data,
                  std::vector<Magnum::Vector2> &vertices,
                  const std::vector<Magnum::Vector2i> &edges,
                  const std::vector<float> &edge_lengths,
                  float time)
{
    // check if data is m'kay
    if (data.velocities.size() != vertices.size())
        data.velocities.resize(vertices.size(), Magnum::Vector2(0, 0));
    if (data.forces.size() != vertices.size())
        data.forces.resize(vertices.size());

    // clean up the forces
    for (auto &v : data.forces)
        v = Magnum::Vector2(0, 0);

    // add repulsive forces
    for (size_t i = 1; i < vertices.size(); ++i)
        for (size_t j = 0; j < i; ++j) {
            auto d = vertices[j] - vertices[i];
            auto q = exp(-d.length() / 50) * 1000 * time;
            data.forces[i] += -q * d;
            data.forces[j] += q * d;
        }

    // add compulsive forces
    for (size_t i = 0; i < edges.size(); ++i) {
        auto p1 = edges[i][0], p2 = edges[i][1];
        auto d = vertices[p2] - vertices[p1];
        auto q = (edge_lengths[i] - d.length()) * 5 * time;
        data.forces[p1] += -q * d;
        data.forces[p2] += q * d;
    }

    // update velocities and positions
    auto slowdown = pow(0.1f, time);
    for (size_t i = 0; i < vertices.size(); ++i) {
        data.velocities[i] += data.forces[i] * time;
        vertices[i] += data.velocities[i] * time;
        data.velocities[i] *= slowdown;
    }
}
