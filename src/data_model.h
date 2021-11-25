
#ifndef DATA_MODEL_H
#define DATA_MODEL_H

#include <string>
#include <vector>

#include "dirty.h"

struct DataModel : public Dirts
{
    std::vector<float> data; // array of structures
    std::vector<std::string> names;
    size_t d;

    DataModel() { clear(); }

    void clear()
    {
        d = n = 0;
        data.clear();
        names.clear();
        touch();
    }
};

#endif
