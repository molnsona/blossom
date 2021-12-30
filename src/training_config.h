/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
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

#ifndef TRAINING_CONFIG_H
#define TRAINING_CONFIG_H

/**
 * @brief Storage of the dynamic parameters of the algorithms that are set in
 * the GUI by user.
 *
 */
struct TrainingConfig
{
    /** Alpha value for SOM algorithm. */
    float som_alpha;
    /** Alpha value for kmeans algorithm. */
    float kmeans_alpha;
    /** Sigma value for SOM algorithm. */
    float sigma;
    /** Gravity value for kmeans algorithm. */
    float gravity;

    /** Number of iterations value for SOM algorithm. */
    int som_iters;
    /** Number of iterations value for kmeans algorithm. */
    int kmeans_iters;

    /** k-neighbors value for generating knn graph algorithm. */
    int kns;
    /** k-neighbors value for t-SNE algorithm. */
    int tsne_k;

    /** Landmark neighborhood size value for EmbedSOM algorithm. */
    int topn;
    /** Boost value for EmbedSOM algorithm. */
    float boost;
    /** Adjust value for EmbedSOM algorithm. */
    float adjust;

    /** Flag that indicates if the kmeans algorithm should be used. */
    bool kmeans_landmark;
    /** Flag that indicates if the SOM algorithm should be used. */
    bool som_landmark;
    /** Flag that indicates if the kNN graph should be generated. */
    bool knn_edges;
    /** Flag that indicates if the graph layout algorithm should be used. */
    bool graph_layout;
    /** Flag that indicates if the t-SNE algorithm should be used. */
    bool tsne_layout;

    /**
     * @brief Calls @ref reset_data().
     *
     */
    TrainingConfig() { reset_data(); }

    /**
     * @brief Resets values to their default values.
     *
     */
    void reset_data()
    {
        som_alpha = kmeans_alpha = 0.001f;
        sigma = 1.1f;
        gravity = 0.01f;
        som_iters = kmeans_iters = 100;
        kns = 3;
        tsne_k = 3;
        topn = 10;
        boost = 2.0f;
        adjust = 0.2f;

        kmeans_landmark = knn_edges = graph_layout = tsne_layout = false;
        som_landmark = true;
    }
};

#endif // #ifndef TRAINING_CONFIG_H
