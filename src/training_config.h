#ifndef TRAINING_CONFIG_H
#define TRAINING_CONFIG_H

struct TrainingConfig
{
    float som_alpha;
    float kmeans_alpha;
    float sigma;
    float gravity;

    int som_iters;
    int kmeans_iters;
    int kns;
    int tsne_k;
    int topn;
    float boost;
    float adjust;

    bool kmeans_landmark;
    bool som_landmark;
    bool knn_edges;
    bool graph_layout;
    bool tsne_layout;

    TrainingConfig() { init(); }

    void init() { reset_data(); }

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
