
#ifndef SCATTER_MODEL_H
#define SCATTER_MODEL_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <vector>

#include "dirty.h"
#include "landmark_model.h"
#include "trans_data.h"

struct ScatterModel : public Sweeper
{
    std::vector<Magnum::Vector2> points;

    Cleaner lm_watch;
    void update(const TransData &d, const LandmarkModel &lm);
};

#endif
