
#include "scatter_model.h"

ScatterModel::ScatterModel()
{
    points.reserve(100);
    for (size_t i = 0; i < 10; ++i)
        for (size_t j = 0; j < 10; ++j)
            points.emplace_back(i / 10.0, j / 10.0);
}
