
#include "knn_edges.h"

#include <cmath>
#include <map>
#include <set>

constexpr float
sqr(float x)
{
    return x * x;
}

void
make_knn_edges(KnnEdgesData &data, LandmarkModel &landmarks, const size_t kns)
{
    /* TODO this might become pretty computation heavy. It might be better to
     * be able to compute just several points in each frame to save framerate,
     * and continuously rotate over the points. */

    landmarks.edges.clear();
    landmarks.edge_lengths.clear();

    if (!(landmarks.n_landmarks() && kns)) {
        return;
    }

    landmarks.edges.reserve((1 + kns) * landmarks.n_landmarks());
    landmarks.edge_lengths.reserve(landmarks.edges.size());

    std::map<std::pair<size_t, size_t>, float> nn;

#if 1
    std::vector<std::pair<float, size_t>> inn(kns + 1);
    for (size_t i = 0; i < landmarks.n_landmarks(); ++i) {
        size_t nns = 0;
        for (size_t j = 0; j < landmarks.n_landmarks(); ++j) {
            if (i == j)
                continue;
            float sqd = 0;
            for (size_t di = 0; di < landmarks.d; ++di)
                sqd += sqr(landmarks.hidim_vertices[di + landmarks.d * i] -
                           landmarks.hidim_vertices[di + landmarks.d * j]);

            if (nns && inn[nns - 1].first <= sqd)
                continue;
            inn[nns] = { sqd, j };
            for (size_t ni = nns; ni > 0; --ni) {
                if (inn[ni].first < inn[ni - 1].first)
                    inn[ni].swap(inn[ni - 1]);
                else
                    break;
            }
            if (nns < kns)
                ++nns;
        }

        for (size_t ni = 0; ni < nns; ++ni)
            nn[{ std::min(i, inn[ni].second), std::max(i, inn[ni].second) }] =
              sqrt(inn[ni].second);
    }
#endif

#if 1
    // add a MST graph to keep connections
    std::vector<bool> visited(landmarks.n_landmarks(), false);
    std::set<std::tuple<float, size_t, size_t>> q;
    q.insert({0, 0, 0});
    while(!q.empty()) {
        auto [curdist, cur, from] = *q.begin();
        q.erase(q.begin());

        if(visited[cur]) continue;
        visited[cur]=true;

        if(cur != from) nn[{std::min(cur,from), std::max(cur,from)}]=curdist;

        for(size_t i=0; i<landmarks.n_landmarks(); ++i) {
            if(visited[i]) continue;
            float sqd = 0;
            for (size_t di = 0; di < landmarks.d; ++di)
                sqd += sqr(landmarks.hidim_vertices[di + landmarks.d * cur] -
                           landmarks.hidim_vertices[di + landmarks.d * i]);

            q.insert({sqd, i, cur});
        }
    }
#endif

    for (auto &&p : nn) {
        landmarks.edges.push_back(p.first);
        landmarks.edge_lengths.push_back(0.05 * sqrt(p.second)); // TODO
    }
}
