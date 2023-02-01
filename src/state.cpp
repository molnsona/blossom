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

#define DEBUG

void
State::update(float actual_time, bool vert_pressed, int vert_ind)
{
    // avoid simulation explosions on long frames
    float time = actual_time;
    if (time > 0.05)
        time = 0.05;

    stats.update(data);
#ifdef DEBUG
    frame_stats.timer.tick();
#endif
    trans.update(data, stats, frame_stats);
#ifdef DEBUG    
    frame_stats.timer.tick();
    if(trans.dim() > 0)
        if(frame_stats.trans_times.size() < 50)
            frame_stats.trans_times.emplace_back(frame_stats.timer.frametime);    

    frame_stats.timer.tick();
#endif
    scaled.update(trans, frame_stats);
#ifdef DEBUG
    frame_stats.timer.tick();
    if(scaled.dim() > 0)
        if(frame_stats.scaled_times.size() < 50)
            frame_stats.scaled_times.emplace_back(frame_stats.timer.frametime);    
#endif

    // TODO only run this on data reset, ideally from trans or from a common
    // trigger
// #ifdef DEBUG // TODO remove
//     landmarks.update_dim(3);
// #else
    landmarks.update_dim(scaled.dim());
// #endif

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
#ifdef DEBUG                        
    frame_stats.timer.tick();
#endif
    colors.update(trans, frame_stats);
#ifdef DEBUG
    frame_stats.timer.tick();
    if(scaled.dim() > 0)
        if(frame_stats.color_times.size() < 50)
            frame_stats.color_times.emplace_back(frame_stats.timer.frametime);    

    frame_stats.timer.tick();
#endif
    scatter.update(scaled, landmarks, training_conf, frame_stats);
#ifdef DEBUG
    frame_stats.timer.tick();
    if(scaled.dim() > 0)                
        frame_stats.scatter_t = frame_stats.timer.frametime 
            * 1000; // to get milliseconds    
#endif
}
