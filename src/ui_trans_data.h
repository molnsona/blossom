#ifndef UI_TRANS_DATA_H
#define UI_TRANS_DATA_H

#include <string>
#include <vector>

struct UiTransData
{
    // Scale factor of each parameter.
    std::vector<float> scale;
    std::vector<std::string> param_names;
    bool scale_mean;
    bool scale_var;
    bool data_changed;
    bool mean_changed;
    bool var_changed;
    bool sliders_changed;
    // Which slider has changed.
    std::vector<bool> sliders;

    UiTransData() { init(); }

    void init() { reset_data(); }

    void reset_data()
    {
        scale.clear();
        param_names.clear();
        scale_mean = scale_var = data_changed = mean_changed = var_changed =
          sliders_changed = false;
        sliders.clear();
    }

    void reset_flags()
    {
        data_changed = mean_changed = var_changed = sliders_changed = false;
        std::fill(sliders.begin(), sliders.end(), false);
    }
};

#endif // #ifndef UI_TRANS_DATA_H
