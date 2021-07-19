
#include "data_model.h"

#include <random>

DataModel::DataModel()
  : n(1000)
  , d(2)
{

    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    data.resize(n * d);
    for (size_t ni = 0; ni < n; ++ni)
        for (size_t di = 0; di < d; ++di)
            data[ni * d + di] = dist(gen);
}
