
#include "tsne_layout.h"

constexpr float
sqrf(float x)
{
    return x * x;
}

void
tsne_layout_step(TSNELayoutData &data,
                 const MouseData &mouse,
                 LandmarkModel &lm,
                 float time)
{
    size_t n = lm.n_landmarks(), d = lm.d;

    if (!n || !d)
        return;

    auto &pji = data.pji;
    auto &heap = data.heap;

    if (pji.size() != n * n)
        pji.resize(n * n);
    if (heap.size() != n)
        heap.resize(n);

    for (size_t i = 0; i < n; ++i) {
        pji[n * i + i] = 0;
        for (size_t j = i + 1; j < n; ++j) {
            float tmp = 0;
            for (size_t di = 0; di < d; ++di)
                tmp += sqrf(lm.hidim_vertices[i * d + di] -
                            lm.hidim_vertices[j * d + di]);
            pji[n * i + j] = tmp;
            pji[n * j + i] = tmp;
        }
    }

    auto hat = [&pji, &heap, n](size_t i, size_t row) -> float {
        return pji[row * n + heap[i]];
    };
    auto hsw = [&heap](size_t i, size_t j) { std::swap(heap[i], heap[j]); };
    auto hdown = [&hat, &hsw](size_t i, size_t n, size_t row) {
        for (;;) {
            size_t l = 2 * i + 1;
            size_t r = l + 1;
            if (l >= n)
                break;
            if (r >= n) {
                if (hat(i, row) > hat(l, row))
                    hsw(i, l);
                break;
            }
            if (hat(l, row) < hat(r, row)) {
                hsw(i, l);
                i = l;
            } else {
                hsw(i, r);
                i = r;
            }
        }
    };
    auto heapify = [&hdown](size_t n, size_t row) {
        size_t i = n / 2 + 1;
        while (i-- > 0)
            hdown(i, n, row);
    };
    auto hpop = [&hdown, &heap](size_t &n, size_t row) -> size_t {
        size_t out = heap[0];
        --n;
        heap[0] = heap[n];
        hdown(0, n, row);
        return out;
    };

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n - 1; ++j)
            heap[j] = j < i ? j : j + 1;
        heapify(n - 1, i);
        size_t hs = n - 1;
        float wsum = 0; // it should sum to 1
        // Dmitry Kobak's slides say this approximation is OK
        for (size_t k = 1; hs; ++k)
            wsum += pji[i * n + hpop(hs, i)] = 1 / float(k);
        // hopefully this code isn't perplexed.
        wsum += 0.001;
        wsum = 1 / wsum;
        for (size_t j = 0; j < n; ++j)
            pji[i * n + j] *= wsum;
    }

    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            pji[i * n + j] = pji[j * n + i] =
              (pji[i * n + j] + pji[j * n + i]) / (2 * n);

    float Z = 0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            Z += 2 / (1 + (lm.lodim_vertices[i] - lm.lodim_vertices[j]).dot());

    Z = 1 / Z;

    auto &ups = data.updates;
    if (ups.size() != n)
        ups.resize(n);
    for (auto &u : ups)
        u = Vector2(0, 0);

    float update_weight = 0;

    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            auto vji = lm.lodim_vertices[i] - lm.lodim_vertices[j];
            float wij =
              1 / (1 + (lm.lodim_vertices[i] - lm.lodim_vertices[j]).dot());
            auto a = vji * pji[i * n + j] * wij;
            ups[i] -= a;
            ups[j] += a;
            update_weight += a.length();
            a = vji * Z * wij * wij;
            ups[i] += a;
            ups[j] -= a;
            update_weight += a.length();
        }

    update_weight = 100 / update_weight;

    for (size_t i = 0; i < n; ++i)
        if (!mouse.vert_pressed || mouse.vert_ind != i)
            lm.lodim_vertices[i] += update_weight * time * ups[i];

    lm.touch();
}
