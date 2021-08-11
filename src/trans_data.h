#ifndef TRANS_DATA_H
#define TRANS_DATA_H

#include <thread>
#include <vector>

#include "data_model.h"
#include "ui_trans_data.h"

struct TransData
{
    size_t d, n;

    TransData() = delete;
    TransData(const std::vector<float> &d, size_t dim, size_t size);

    void set_data(const std::vector<float> &dat, size_t dim, size_t size)
    {
        data = dat;
        d = dim;
        n = size;
    }
    const std::vector<float> &get_data() { return data; }
    void update(UiTransData &ui, const DataModel &orig_data);

private:
    void fn_wrapper(const UiTransData &ui, const DataModel &orig_data);

    std::vector<float> data;

    UiTransData data_snapshot;
    DataModel orig_data_snapshot;

    std::thread trans_thread;
    bool thread_finished;
};

#endif // #ifndef TRANS_DATA_H
