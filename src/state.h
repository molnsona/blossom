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

#ifndef STATE_H
#define STATE_H

#include <memory>
#include <string>
#include <vector>

#include "color_data.h"
#include "data_model.h"
#include "graph_layout.h"
#include "kmeans_landmark.h"
#include "knn_edges.h"
#include "landmark_model.h"
#include "mouse_data.h"
#include "scaled_data.h"
#include "scatter_model.h"
#include "training_config.h"
#include "trans_data.h"
#include "tsne_layout.h"

/**
 * @brief Storage of data of used algorithms and input events.
 *
 * It represents state of the simulation in current frame and performs steps of
 * the simulation in each frame.
 *
 */
struct State
{
    DataModel data;
    RawDataStats stats;
    TransData trans;
    ScaledData scaled;
    LandmarkModel landmarks;

    TrainingConfig training_conf;
    GraphLayoutData layout_data;
    TSNELayoutData tsne_data;
    KMeansData kmeans_data;
    KnnEdgesData knn_data;

    ColorData colors;
    ScatterModel scatter;

    /**
     * @brief Performs simulation steps of all active algorithms and updates
     * data according to the user interaction.
     *
     * @param time Duration of the last frame.
     */
    void update(float time, const MouseData &mouse);
};

#endif // #ifndef STATE_H
