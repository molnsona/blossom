
#include "embedsom.h"

/* This file is originally a part of EmbedSOM.
 *
 * Copyright (C) 2018-2021 Mirek Kratochvil <exa.exa@gmail.com>
 *
 * EmbedSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * EmbedSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * EmbedSOM. If not, see <https://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

using namespace std;

/*
 * Distance computation tools
 */

static inline float
sqrf(float n)
{
    return n * n;
}

namespace distfs {

struct sqeucl
{
    inline static float back(float x) { return sqrt(x); }
    inline static float comp(const float *p1, const float *p2, const size_t dim)
    {
#ifndef USE_INTRINS
        float sqdist = 0;
        for (size_t i = 0; i < dim; ++i) {
            float tmp = p1[i] - p2[i];
            sqdist += tmp * tmp;
        }
        return sqdist;
#else
        const float *p1e = p1 + dim, *p1ie = p1e - 3;

        __m128 s = _mm_setzero_ps();
        for (; p1 < p1ie; p1 += 4, p2 += 4) {
            __m128 tmp = _mm_sub_ps(_mm_loadu_ps(p1), _mm_loadu_ps(p2));
            s = _mm_add_ps(_mm_mul_ps(tmp, tmp), s);
        }
        float sqdist = s[0] + s[1] + s[2] + s[3];
        for (; p1 < p1e; ++p1, ++p2) {
            float tmp = *p1 - *p2;
            sqdist += tmp * tmp;
        }
        return sqdist;
#endif
    }

    inline static void proj(const float *la,
                            const float *lb,
                            const float *p,
                            const size_t dim,
                            float &o_scalar,
                            float &o_sqdist)
    {

#ifndef USE_INTRINS
        float scalar = 0, sqdist = 0;
        for (size_t k = 0; k < dim; ++k) {
            float tmp = lb[k] - la[k];
            sqdist += tmp * tmp;
            scalar += tmp * (p[k] - la[k]);
        }
#else
        float scalar, sqdist;
        {
            const float *ke = la + dim, *kie = ke - 3;

            __m128 sca = _mm_setzero_ps(), sqd = _mm_setzero_ps();
            for (; la < kie; la += 4, lb += 4, p += 4) {
                __m128 ti = _mm_loadu_ps(la);
                __m128 tmp = _mm_sub_ps(_mm_loadu_ps(lb), ti);
                sqd = _mm_add_ps(sqd, _mm_mul_ps(tmp, tmp));
                sca = _mm_add_ps(
                  sca, _mm_mul_ps(tmp, _mm_sub_ps(_mm_loadu_ps(p), ti)));
            }
            scalar = sca[0] + sca[1] + sca[2] + sca[3];
            sqdist = sqd[0] + sqd[1] + sqd[2] + sqd[3];
            for (; la < ke; ++la, ++lb, ++p) {
                float tmp = *lb - *la;
                sqdist += tmp * tmp;
                scalar += tmp * (*p - *la);
            }
        }
#endif
        o_scalar = scalar;
        o_sqdist = sqdist;
    }
};

} // namespace distfs

/*
 * some required constants
 */

// small numbers first!
static const float min_boost = 1e-5; // lower limit for the parameter

// this affects how steeply the score decreases for near-farthest codes
static const float max_avoidance = 10;

// this is added before normalizing the distances
static const float zero_avoidance = 1e-10;

// a tiny epsilon for preventing singularities
static const float koho_gravity = 1e-5;

/*
 * KNN computation (mainly heap)
 */
struct dist_id
{
    float dist;
    size_t id;
};

static void
heap_down(dist_id *heap, size_t start, size_t lim)
{
    for (;;) {
        size_t L = 2 * start + 1;
        size_t R = L + 1;
        if (R < lim) {
            float dl = heap[L].dist;
            float dr = heap[R].dist;

            if (dl > dr) {
                if (heap[start].dist >= dl)
                    break;
                swap(heap[L], heap[start]);
                start = L;
            } else {
                if (heap[start].dist >= dr)
                    break;
                swap(heap[R], heap[start]);
                start = R;
            }
        } else if (L < lim) {
            if (heap[start].dist < heap[L].dist)
                swap(heap[L], heap[start]);
            break; // exit safely!
        } else
            break;
    }
}

template<class distf>
static void
knn(const float *point,
    const float *hidim_lm,
    size_t n_landmarks,
    size_t dim,
    size_t topnn,
    vector<dist_id> &dists)
{
    size_t i;

    // push first topnn kohos
    for (i = 0; i < topnn; ++i) {
        dists[i].dist = distf::comp(point, hidim_lm + i * dim, dim);
        dists[i].id = i;
    }

    // make a heap
    for (i = 0; i < topnn; ++i)
        heap_down(dists.data(), topnn - i - 1, topnn);

    // insert the rest
    for (i = topnn; i < n_landmarks; ++i) {
        float s = distf::comp(point, hidim_lm + i * dim, dim);
        if (dists[0].dist < s)
            continue;
        dists[0].dist = s;
        dists[0].id = i;
        heap_down(dists.data(), 0, topnn);
    }

    // heapsort the NNs
    for (i = topnn - 1; i > 0; --i) {
        swap(dists[0], dists[i]);
        heap_down(dists.data(), 0, i);
    }
}

/*
 * Projection- and fitting-related helpers
 */

template<int embed_dim>
static void
add_gravity(const float *lodim_lm, float score, float *mtx)
{
    float gs = score * koho_gravity;
    if (embed_dim == 2) {
        mtx[0] += gs;
        mtx[3] += gs;
        mtx[4] += gs * lodim_lm[0];
        mtx[5] += gs * lodim_lm[1];
    }
    if (embed_dim == 3) {
        mtx[0] += gs;
        mtx[4] += gs;
        mtx[8] += gs;
        mtx[9] += gs * lodim_lm[0];
        mtx[10] += gs * lodim_lm[1];
        mtx[11] += gs * lodim_lm[2];
    }
}

template<int embed_dim>
inline static float
dotp_ec(const float *a, const float *b)
{
    float r = 0;
    for (size_t i = 0; i < embed_dim; ++i)
        r += a[i] * b[i];
    return r;
}

template<int embed_dim>
static void
add_approximation(float score_i,
                  float score_j,
                  const float *ilm,
                  const float *jlm,
                  float scalar_proj,
                  float adjust,
                  float *mtx)
{
    float h[embed_dim], hp = 0;
    for (size_t i = 0; i < embed_dim; ++i)
        hp += sqrf(h[i] = jlm[i] - ilm[i]);
    if (hp < zero_avoidance)
        return;

    const float s =
      score_i * score_j * powf(1 + hp, -adjust) * expf(-sqrf(scalar_proj - .5));
    const float sihp = s / hp;
    const float rhsc = s * (scalar_proj + dotp_ec<embed_dim>(h, ilm) / hp);

    if (embed_dim == 2) {

        mtx[0] += h[0] * h[0] * sihp;
        mtx[1] += h[0] * h[1] * sihp;
        mtx[2] += h[1] * h[0] * sihp;
        mtx[3] += h[1] * h[1] * sihp;
        mtx[4] += h[0] * rhsc;
        mtx[5] += h[1] * rhsc;
    }

    if (embed_dim == 3) {
        mtx[0] += h[0] * h[0] * sihp;
        mtx[1] += h[0] * h[1] * sihp;
        mtx[2] += h[0] * h[2] * sihp;
        mtx[3] += h[1] * h[0] * sihp;
        mtx[4] += h[1] * h[1] * sihp;
        mtx[5] += h[1] * h[2] * sihp;
        mtx[6] += h[2] * h[0] * sihp;
        mtx[7] += h[2] * h[1] * sihp;
        mtx[8] += h[2] * h[2] * sihp;
        mtx[9] += h[0] * rhsc;
        mtx[10] += h[1] * rhsc;
        mtx[11] += h[2] * rhsc;
    }
}

template<int embed_dim>
static void
solve_lin_eq(const float *mtx, float *embedding)
{
    if (embed_dim == 2) {
        float det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
        embedding[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
        embedding[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
    }
    if (embed_dim == 3) {
        // this looks ugly
        float det = mtx[0] * mtx[4] * mtx[8] + mtx[1] * mtx[5] * mtx[6] +
                    mtx[2] * mtx[3] * mtx[7] - mtx[0] * mtx[5] * mtx[7] -
                    mtx[1] * mtx[3] * mtx[8] - mtx[2] * mtx[4] * mtx[6];
        embedding[0] = (mtx[9] * mtx[4] * mtx[8] + mtx[10] * mtx[5] * mtx[6] +
                        mtx[11] * mtx[3] * mtx[7] - mtx[9] * mtx[5] * mtx[7] -
                        mtx[10] * mtx[3] * mtx[8] - mtx[11] * mtx[4] * mtx[6]) /
                       det;
        embedding[1] = (mtx[0] * mtx[10] * mtx[8] + mtx[1] * mtx[11] * mtx[6] +
                        mtx[2] * mtx[9] * mtx[7] - mtx[0] * mtx[11] * mtx[7] -
                        mtx[1] * mtx[9] * mtx[8] - mtx[2] * mtx[10] * mtx[6]) /
                       det;
        embedding[2] = (mtx[0] * mtx[4] * mtx[11] + mtx[1] * mtx[5] * mtx[9] +
                        mtx[2] * mtx[3] * mtx[10] - mtx[0] * mtx[5] * mtx[10] -
                        mtx[1] * mtx[3] * mtx[11] - mtx[2] * mtx[4] * mtx[9]) /
                       det;
    }
}

/*
 * scoring
 */
template<class distf>
static void
sorted_dists_to_scores(vector<dist_id> &dists,
                       const size_t topn,
                       const size_t topnn,
                       const float boost)
{
    // compute the distance distribution for the scores
    float mean = 0, sd = 0, wsum = 0;
    for (size_t i = 0; i < topnn; ++i) {
        const float tmp = distf::back(dists[i].dist);
        const float w = 1 / float(i + 1);
        mean += tmp * w;
        sd += tmp * tmp * w;
        wsum += w;
        dists[i].dist = tmp;
    }

    mean /= wsum;
    sd = boost / sqrtf(sd / wsum - sqrf(mean));
    const float nmax = max_avoidance / dists[topnn - 1].dist;

    // convert the stuff to scores
    for (size_t i = 0; i < topn; ++i)
        if (topn < topnn)
            dists[i].dist = expf((mean - dists[i].dist) * sd) *
                            (1 - expf(dists[i].dist * nmax - max_avoidance));
        else
            dists[i].dist = expf((mean - dists[i].dist) * sd);
}

/*
 * EmbedSOM function for a single point
 */
template<class distf, int embed_dim>
static void
embedsom_point(const size_t n_landmarks,
               const size_t dim,
               const float boost,
               const size_t topn,
               const float adjust,
               const float *point,
               const float *hidim_lm,
               const float *lodim_lm,
               float *embedding,
               vector<dist_id> &dists)
{
    const size_t topnn = topn < n_landmarks ? topn + 1 : topn;

    knn<distf>(point, hidim_lm, n_landmarks, dim, topnn, dists);

    sorted_dists_to_scores<distf>(dists, topn, topnn, boost);

    // create the empty equation matrix
    float mtx[embed_dim * (1 + embed_dim)];
    fill(mtx, mtx + embed_dim * (1 + embed_dim), 0);

    // for all points in the neighborhood
    for (size_t i = 0; i < topn; ++i) {
        size_t idx = dists[i].id;
        float score_i = dists[i].dist; // score of 'i'

        float ilm[embed_dim]; // lodim_lm for point 'i'
        copy_n(lodim_lm + embed_dim * idx, embed_dim, ilm);

        /* this adds a really tiny influence of the point to
         * prevent singularities */
        add_gravity<embed_dim>(ilm, score_i, mtx);

        // for all combinations of point 'i' with points in the
        // neighborhood
        for (size_t j = i + 1; j < topn; ++j) {

            size_t jdx = dists[j].id;
            float score_j = dists[j].dist; // score of 'j'

            float jlm[embed_dim]; // lodim_lm for point 'j'
            copy_n(lodim_lm + embed_dim * jdx, embed_dim, jlm);

            float scalar, sqdist;
            distf::proj(hidim_lm + dim * idx,
                        hidim_lm + dim * jdx,
                        point,
                        dim,
                        scalar,
                        sqdist);

            if (sqdist == 0)
                continue;
            else
                scalar /= sqdist;

            add_approximation<embed_dim>(
              score_i, score_j, ilm, jlm, scalar, adjust, mtx);
        }
    }

    solve_lin_eq<embed_dim>(mtx, embedding);
}

void
embedsom(size_t n,
         size_t n_landmarks,
         size_t dim,
         float boost,
         size_t topn,
         float adjust,
         const float *points,
         const float *hidim_lm,
         const float *lodim_lm,
         float *embedding)
{
    if (topn > n_landmarks)
        topn = n_landmarks;

    const size_t topnn = topn < n_landmarks ? topn + 1 : topn;

    if (n_landmarks < 3)
        return;

    vector<dist_id> dists;
    dists.resize(topnn);

    // TODO constkill the lowdim=2 template parameter
    for (size_t i = 0; i < n; ++i)
        embedsom_point<distfs::sqeucl, 2>(n_landmarks,
                                          dim,
                                          boost,
                                          topn,
                                          adjust,
                                          points + dim * i,
                                          hidim_lm,
                                          lodim_lm,
                                          embedding + 2 * i,
                                          dists);
}
