#include "trans_data.h"

TransData::TransData(const std::vector<float> &d)
  : data(d)
{}

void
TransData::update(UiTransData &ui)
{
    if (!thread_finished)
        return;

    if (trans_thread.joinable())
        trans_thread.join();

    if (ui.data_changed) {
        thread_finished = false;

        UiTransData cpy_ui = ui;
        // Reset sliders and all flags.
        ui.reset_flags();

        trans_thread = std::thread(&TransData::fn_wrapper, // method
                                      this,                   // instance
                                      std::cref(cpy_ui)       // parameter
        );
    }
}

void
TransData::fn_wrapper(const UiTransData &ui)
{

    thread_finished = true;
}
