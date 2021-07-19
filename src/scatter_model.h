
#ifndef SCATTER_MODEL_H
#define SCATTER_MODEL_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <vector>

struct ScatterModel
{
    std::vector<Magnum::Vector2> points;

    ScatterModel();
};

#endif
