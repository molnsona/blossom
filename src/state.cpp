
#include "state.h"
#include "fcs_parser.h"
#include "tsv_parser.h"

State::State() {}

void
State::update(float actual_time)
{
    // avoid simulation explosions on long frames
    float time = actual_time;
    if (time > 0.05)
        time = 0.05;

    stats.update(data);
    trans.update(data, stats);
    scaled.update(trans);

    // TODO only run this on data reset, ideally from trans or from a common
    // trigger
    landmarks.update_dim(scaled.dim());

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
                                           // they they are created.

    if (training_conf.graph_layout)
        graph_layout_step(layout_data, mouse, landmarks, time);

    if (training_conf.tsne_layout)
        tsne_layout_step(tsne_data, mouse, landmarks, time);

    if (training_conf.som_landmark)
        som_landmark_step(kmeans_data,
                          scaled,
                          training_conf.som_iters,
                          training_conf.som_alpha,
                          training_conf.sigma,
                          landmarks);

    colors.update(trans);
    scatter.update(scaled, landmarks, training_conf);
}
