
#include <iostream>

#include "embedsom.h"
#include "fcs_parser.h"
#include "state.h"
#include "tsv_parser.h"

State::State() {}

void
State::update(float actual_time, UiData &ui)
{
    // avoid simulation explosions on long frames
    float time = actual_time;
    if (time > 0.05)
        time = 0.05;

    stats.update(data);
    trans.update(data, stats);

    // TODO only run this on data reset, ideally from trans or from a common
    // trigger
    landmarks.update_dim(trans.dim());

    if (training_conf.kmeans_landmark)
        kmeans_landmark_step(
          kmeans_data,
          trans,
          100,   // TODO parametrize (now this is 100 iters per frame, there
                 // should be fixed number of iters per actual elapsed time)
          0.01,  // TODO parametrize, logarithmically between 1e-6 and ~0.5
          0.001, // TODO parametrize as 0-1 multiple of ^^
          landmarks);

    if (training_conf.knn_edges)
        make_knn_edges(knn_data, landmarks, 3);

    if (training_conf.graph_layout)
        graph_layout_step(layout_data, mouse, landmarks, time);

    if (training_conf.som_landmark)
        som_landmark_step(kmeans_data,
                          trans,
                          100,
                          training_conf.alpha,
                          training_conf.sigma,
                          landmarks);

#ifdef NO_CUDA

    scatter.update(trans, landmarks);

#else
    // these methods should be called only once in the initialization and then
    // only when the data/paramters change
    esom_cuda.setDim(trans.dim());
    esom_cuda.setK(10);
    esom_cuda.setPoints(trans.n, trans.get_data().data());
    esom_cuda.setLandmarks(landmarks.lodim_vertices.size(),
                           landmarks.hidim_vertices.data(),
                           landmarks.lodim_vertices[0].data());

    // this is the actual method that is called in every update
    esom_cuda.embedsom(
      2.0, 0.2, scatter.points[0].data()); // boost and adjust parameters are
                                           // now passed in every call, but we
                                           // might want to cache them iside?

    static std::size_t counter = 0;

    if (++counter >= 10) {
        std::cout << esom_cuda.getAvgPointsUploadTime() << "ms \t"
                  << esom_cuda.getAvgLandmarksUploadTime() << "ms \t"
                  << esom_cuda.getAvgProcessingTime() << "ms" << std::endl;
    }
    counter %= 10;
#endif
}
