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

//#define DEBUG
#include <iostream>
#define MEASURE_CONST(name, method)\
    frame_stats.timer.tick();\
    frame_stats.constant_time +=\
        frame_stats.timer.frametime * 1000;\
    frame_stats.timer.tick();\
    method;\
    frame_stats.timer.tick();\
    frame_stats.constant_time +=\
        frame_stats.timer.frametime * 1000;\
    std::cout << name << frame_stats.timer.frametime * 1000 << std::endl;\
    frame_stats.timer.tick();

#define MEASURE_NON_CONST(name, T, method)\
    frame_stats.timer.tick();\
    frame_stats.constant_time += \
        frame_stats.timer.frametime * 1000;\
    frame_stats.timer.tick();  \
    method;\
    frame_stats.timer.tick();\
    T =\
      frame_stats.timer.frametime * 1000; /* to get milliseconds*/\
    std::cout << name << T << std::endl;\
    frame_stats.timer.tick();

void
State::update(float actual_time, bool vert_pressed, int vert_ind)
{
    // avoid simulation explosions on long frames
    float time = actual_time;
    if (time > 0.05)
        time = 0.05;

    // Compute time for estimation methods
    compute_time(frame_stats);

    stats.update(data);

    // MEASURE_NON_CONST("trans: ", frame_stats.trans_t, 
    //     trans.update(data, stats, frame_stats));
    trans.update(data, stats, frame_stats);
    scaled.update(trans, frame_stats);

    // TODO only run this on data reset, ideally from trans or from a common
    // trigger
#ifdef DEBUG // TODO remove
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

    colors.update(trans, frame_stats);
    scatter.update(scaled, landmarks, training_conf, frame_stats);
}

void State::compute_time(FrameStats& fs)
{    
    // First compute statistics
    // if all 4 are computing
    if(fs.trans_t > 0.00001f && fs.scaled_t > 0.00001f &&
        fs.embedsom_t > 0.00001f && fs.color_t > 0.00001f) {
        fs.trans_priority = 0.375f;
        fs.scaled_priority = 0.375f;
        fs.color_priority = 0.125f;
        fs.embed_priority = 0.125f;        
    } 
    else
    // if trans finished and all other are computing
    if(fs.trans_t <= 0.00001f && fs.scaled_t > 0.00001f &&
        fs.embedsom_t > 0.00001f && fs.color_t > 0.00001f) {
        fs.trans_priority = 0.0f;
        fs.scaled_priority = 0.75f;
        fs.color_priority = 0.125f;
        fs.embed_priority = 0.125f;
    }
    else
    // if scaled finished and all others are computing
    if(fs.trans_t > 0.00001f && fs.scaled_t <= 0.00001f &&
        fs.embedsom_t > 0.00001f && fs.color_t > 0.00001f) {
        fs.trans_priority = 0.75f;
        fs.scaled_priority = 0.0f;
        fs.color_priority = 0.125f;
        fs.embed_priority = 0.125f;
    }
    else
    // if trans and scaled finished computing
    if(fs.trans_t <= 0.00001f && fs.scaled_t <= 0.00001f &&
        fs.embedsom_t > 0.00001f && fs.color_t > 0.00001f) {
        fs.trans_priority = 0.0f;
        fs.scaled_priority = 0.0f;
        fs.color_priority = 0.5f;
        fs.embed_priority = 0.5f;
    }
    else
    // if trans, scaled and color finished computing
    if(fs.trans_t <= 0.00001f && fs.scaled_t <= 0.00001f &&
        fs.embedsom_t > 0.00001f && fs.color_t <= 0.00001f) {
        fs.trans_priority = 0.0f;
        fs.scaled_priority = 0.0f;
        fs.color_priority = 0.0f;
        fs.embed_priority = 1.0f;
    }
    else
    // if trans, scaled and embedsom finished computing
    if(fs.trans_t <= 0.00001f && fs.scaled_t <= 0.00001f &&
        fs.embedsom_t <= 0.00001f && fs.color_t > 0.00001f) {
        fs.trans_priority = 0.0f;
        fs.scaled_priority = 0.0f;
        fs.color_priority = 1.0f;
        fs.embed_priority = 0.0f;
    }
    else
    // if trans, scaled, color and embedsom finished computing
    if(fs.trans_t <= 0.00001f && fs.scaled_t <= 0.00001f &&
        fs.embedsom_t <= 0.00001f && fs.color_t <= 0.00001f) {
        fs.trans_priority = 0.0f;
        fs.scaled_priority = 0.0f;
        fs.color_priority = 0.0f;
        fs.embed_priority = 0.0f;
    }

    float alpha = 0.05f;
    float coalpha = 1 - 0.05f;
    if(fs.trans_priority == 0.0f) fs.trans_duration = 0.0f;
    else fs.trans_duration =
        fs.trans_duration * coalpha + 
        fs.est_time * fs.trans_priority * alpha;
    if(fs.embed_priority == 0.0f) fs.embedsom_duration = 0.0f;
    else fs.embedsom_duration =
        fs.embedsom_duration * coalpha +
        fs.est_time * fs.embed_priority * alpha;
    if(fs.scaled_priority == 0.0f) fs.scaled_duration = 0.0f;
    else fs.scaled_duration =
        fs.scaled_duration * coalpha +
        fs.est_time * fs.scaled_priority * alpha;
    if(fs.color_priority == 0.0f) fs.color_duration = 0.0f;
    else fs.color_duration = 
        fs.color_duration * coalpha + 
        fs.est_time * fs.color_priority * alpha;       
}
