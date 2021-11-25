#ifndef TRAINING_CONFIG_H
#define TRAINING_CONFIG_H

struct TrainingConfig
{
    float alpha;
    float sigma;

    TrainingConfig() { init(); }

    void init() { reset_data(); }

    void reset_data()
    {
        alpha = 0.001f;
        sigma = 1.1f;
    }
};

#endif // #ifndef TRAINING_CONFIG_H
