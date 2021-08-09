#ifndef TRANS_DATA_H
#define TRANS_DATA_H

#include <vector>

struct TransData
{
    TransData() = delete;
    TransData(const std::vector<float> &d);

    void update(const std::vector<float> &d) { data = d; }

private:
    std::vector<float> data;
};

#endif // #ifndef TRANS_DATA_H
