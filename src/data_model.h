
#ifndef DATA_MODEL_H
#define DATA_MODEL_H

#include <string>
#include <vector>

struct DataModel
{
    std::vector<float> data; // array of structures
    std::vector<std::string> names;
    size_t d, n;

    DataModel() { clear(); }

    void clear()
    {
        data.clear();
        names.clear();
        d = n = 0;
    }
};

#endif
