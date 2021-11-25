#ifndef TRAINING_CONFIG_H
#define TRAINING_CONFIG_H

struct TrainingConfig
{
    float alpha;
    float sigma;
    bool kmeans_landmark;
    bool som_landmark;
    bool knn_edges;
    bool graph_layout;

    TrainingConfig() { init(); }

    void init() { reset_data(); }

    void reset_data()
    {
        kmeans_landmark = knn_edges = graph_layout = false;
        som_landmark = true;
        alpha = 0.001f;
        sigma = 1.1f;
    }
};

#endif // #ifndef TRAINING_CONFIG_H
