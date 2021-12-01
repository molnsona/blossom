
#ifndef SCATTER_MODEL_H
#define SCATTER_MODEL_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <vector>

#include "dirty.h"
#include "landmark_model.h"
#include "scaled_data.h"
#include "training_config.h"

struct ScatterModel : public Sweeper
{
    std::vector<Magnum::Vector2> points;

    Cleaner lm_watch;
    void update(const ScaledData &d,
                const LandmarkModel &lm,
                const TrainingConfig &tc);

    void touch_config() { refresh(points.size()); }
};

#endif
