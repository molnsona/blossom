
#include "kmeans_landmark.h"

#include <limits>

constexpr float
sqr(float x)
{
    return x * x;
}

void
kmeans_landmark_step(KMeansData &data,
                     const DataModel &model,
                     size_t n_means,
                     size_t d,
                     size_t iters,
                     float alpha,
                     float neighbor_alpha,
                     const std::vector<std::pair<size_t, size_t>> &neighbors,
                     std::vector<float> &means)
{
    if (d != model.d || means.size() != n_means * d)
        return; // TODO throw something useful

    if (!(model.n && n_means))
        return; // this might happen but it's technically okay

    std::uniform_int_distribution<size_t> random_target(0, model.n - 1);

    for (size_t iter = 0; iter < iters; ++iter) {
        size_t tgt = random_target(data.gen);

        size_t best = 0;
        float best_sqdist = std::numeric_limits<float>::infinity();

        // find the mean that's closest to the target
        for (size_t mi = 0; mi < n_means; ++mi) {
            float sqd = 0;

            for (size_t di = 0; di < d; ++di)
                sqd += sqr(means[di + d * mi] - model.data[di + d * tgt]);

            if (sqd < best_sqdist) {
                best = mi;
                best_sqdist = sqd;
            }
        }

        // move the mean a bit closer
        for (size_t di = 0; di < d; ++di)
            means[di + d * best] +=
              alpha * (model.data[di + d * tgt] - means[di + d * best]);

        // also move the neighbors a bit
        size_t moved = 0;
        for (size_t ei = 0; ei < neighbors.size(); ++ei) {
            size_t nb;
            if (neighbors[ei].first == best)
                nb = neighbors[ei].second;
            else if (neighbors[ei].second == best)
                nb = neighbors[ei].first;
            else
                continue;
            ++moved;
            for (size_t di = 0; di < d; ++di)
                means[di + d * nb] +=
                  neighbor_alpha *
                  (model.data[di + d * tgt] - means[di + d * nb]);
        }
    }
}
