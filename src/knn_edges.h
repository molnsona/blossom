
#ifndef KNN_EDGES_H
#define KNN_EDGES_H

#include "landmark_model.h"

struct KnnEdgesData
{
    // actually not used now
    size_t last_point;

    KnnEdgesData()
      : last_point(0)
    {}
};

void
make_knn_edges(KnnEdgesData &data, LandmarkModel &landmarks, size_t kns);

#endif
