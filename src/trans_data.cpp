#include "trans_data.h"

TransData::TransData(const std::vector<float> &d, size_t dim, size_t size)
  : d(dim)
  , n(size)
  , data(d)
{}

void
TransData::update(UiTransData &ui, const DataModel &orig_data)
{
    if (!thread_finished)
        return;

    if (trans_thread.joinable())
        trans_thread.join();

    if (ui.data_changed) {
        thread_finished = false;

        data_snapshot = ui;
        orig_data_snapshot = orig_data;
        // Reset sliders and all flags.
        ui.reset_flags();

        trans_thread = std::thread(&TransData::fn_wrapper,   // method
                                   this,                     // instance
                                   std::cref(data_snapshot), // parameters
                                   std::cref(orig_data_snapshot));
    }
}

void
TransData::fn_wrapper(const UiTransData &ui, const DataModel &orig_data)
{
    if (ui.mean_changed) {
        if (ui.scale_mean) {
            // scale mean to 0
        }
    }

    if (ui.var_changed) {
        if (ui.scale_var) {
            // scale var to 1
        }
    }

    if (ui.sliders_changed) {
        for (size_t i = 0; i < d; ++i) {
            if (ui.sliders[i]) {
                // multiply all points in given dimension
                for (size_t j = 0; j < n; ++j) {
                    // TODO: check correct dimensions and sizes of original and
                    // transformed data.
                    data[j * d + i] = orig_data.data[j * d + i] * ui.scale[i];
                }
            }
        }
    }

    thread_finished = true;
}
