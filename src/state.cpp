/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Martin Krulis
 *                    Mirek Kratochvil
 *                    Sona Molnarova
 *
 * BlosSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * BlosSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * BlosSOM. If not, see <https://www.gnu.org/licenses/>.
 */

#include "state.h"
#include "fcs_parser.h"
#include "tsv_parser.h"

void
State::update(float actual_time, bool vert_pressed, int vert_ind)
{
    // avoid simulation explosions on long frames
    float time = actual_time;
    if (time > 0.05)
        time = 0.05;

    // Compute time for estimation batch sizes computations.
    frame_stats.update_times();

    stats.update(data);

    trans.update(data, stats, frame_stats);
    scaled.update(trans, frame_stats);

    // TODO only run this on data reset, ideally from trans or from a common
    // trigger
#ifdef STATE_DEBUG // TODO remove
    landmarks.update_dim(3);
#else
    landmarks.update_dim(scaled.dim());
#endif

    if (training_conf.kmeans_landmark)
        kmeans_landmark_step(kmeans_data,
                             scaled,
                             training_conf.kmeans_iters,
                             training_conf.kmeans_alpha,
                             training_conf.gravity,
                             landmarks);

    if (training_conf.knn_edges)
        make_knn_edges(knn_data,
                       landmarks,
                       training_conf.kns); // TODO: Edges are not removed, once
                                           // they are created.

    if (training_conf.graph_layout)
        graph_layout_step(layout_data, vert_pressed, vert_ind, landmarks, time);

    if (training_conf.tsne_layout)
        tsne_layout_step(tsne_data, vert_pressed, vert_ind, landmarks, time);

    if (training_conf.som_landmark)
        som_landmark_step(kmeans_data,
                          scaled,
                          training_conf.som_iters,
                          training_conf.som_alpha,
                          training_conf.sigma,
                          landmarks);

    colors.update(trans, landmarks, frame_stats);
    scatter.update(scaled, landmarks, training_conf, frame_stats);
}
