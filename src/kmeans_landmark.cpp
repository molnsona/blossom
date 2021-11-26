
#include "kmeans_landmark.h"

#include <limits>

constexpr float
sqr(float x)
{
    return x * x;
}

void
kmeans_landmark_step(KMeansData &data,
                     const ScaledData &model,
                     size_t iters,
                     float alpha,
                     float gravity,
                     LandmarkModel &lm)
{
    size_t n_means = lm.n_landmarks();
    size_t d = lm.d;
    auto &means = lm.hidim_vertices;

    gravity *= alpha / n_means;

    if (!(model.n && n_means))
        return; // this might happen but it's technically okay

    if (d != model.dim() || means.size() != n_means * d)
        return; // TODO throw something useful

    std::uniform_int_distribution<size_t> random_target(0, model.n - 1);

    lm.touch();
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

        // move the means a bit closer
        for (size_t mi = 0; mi < n_means; ++mi)
            for (size_t di = 0; di < d; ++di)
                means[di + d * best] +=
                  (mi == tgt ? alpha : gravity) *
                  (model.data[di + d * mi] - means[di + d * best]);
    }
}

void
som_landmark_step(KMeansData &data,
                  const ScaledData &model,
                  size_t iters,
                  float alpha,
                  float sigma,
                  LandmarkModel &lm)
{
    size_t n_neurons = lm.n_landmarks();
    size_t d = lm.d;
    const auto &map = lm.lodim_vertices;
    auto &neurons = lm.hidim_vertices;

    const float nisqsigma = -1.0 / (sigma * sigma);

    if (!(model.n && n_neurons))
        return; // this might happen but it's technically okay

    if (d != model.dim() || neurons.size() != n_neurons * d ||
        map.size() != n_neurons)
        return; // TODO throw something useful

    std::uniform_int_distribution<size_t> random_target(0, model.n - 1);

    lm.touch();
    for (size_t iter = 0; iter < iters; ++iter) {
        size_t tgt = random_target(data.gen);

        size_t best = 0;
        float best_sqdist = std::numeric_limits<float>::infinity();

        // find the mean that's closest to the target
        for (size_t ni = 0; ni < n_neurons; ++ni) {
            float sqd = 0;

            for (size_t di = 0; di < d; ++di)
                sqd += sqr(neurons[di + d * ni] - model.data[di + d * tgt]);

            if (sqd < best_sqdist) {
                best = ni;
                best_sqdist = sqd;
            }
        }

        // move the rest according to the neighborhood function
        for (size_t ni = 0; ni < n_neurons; ++ni) {
            float r = alpha * exp((map[best] - map[ni]).dot() * nisqsigma);

            for (size_t di = 0; di < d; ++di)
                neurons[di + d * ni] +=
                  r * (model.data[di + d * tgt] - neurons[di + d * ni]);
        }
    }
}
