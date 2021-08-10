#ifndef TRANS_DATA_H
#define TRANS_DATA_H

#include <thread>
#include <vector>

#include "ui_trans_data.h"

struct TransData
{
    TransData() = delete;
    TransData(const std::vector<float> &d);

    void set_data(const std::vector<float> &d) { data = d; }
    void update(UiTransData &ui);

private:
    void fn_wrapper(const UiTransData &ui);

    std::vector<float> data;

    std::thread trans_thread;
    bool thread_finished{ true };
};

#endif // #ifndef TRANS_DATA_H
