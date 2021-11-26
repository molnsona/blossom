#ifndef TRAINING_CONFIG_H
#define TRAINING_CONFIG_H

struct TrainingConfig
{
    float alpha;
    float sigma;
    float gravity;

    int iters;
    int kns;
    int topn;
    float boost;
    float adjust;

    bool kmeans_landmark;
    bool som_landmark;
    bool knn_edges;
    bool graph_layout;

    TrainingConfig() { init(); }

    void init() { reset_data(); }

    void reset_data()
    {
        alpha = 0.001f;
        sigma = 1.1f;
        gravity = 0.01f;
        iters = 100;
        kns = 3;
        topn = 10;
        boost = 2.0f;
        adjust = 0.2f;

        kmeans_landmark = knn_edges = graph_layout = false;
        som_landmark = true;
    }
};

#endif // #ifndef TRAINING_CONFIG_H
