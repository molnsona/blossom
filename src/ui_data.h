#ifndef UI_DATA_H
#define UI_DATA_H

#include <memory>
#include <string>
#include <vector>

struct UiData
{
    // Scale factor of each parameter.
    std::vector<float> scale;
    std::vector<std::string> param_names;
    bool scale_mean{ false };
    bool scale_var{ false };
    bool data_changed{ false };

    bool parse{ false };
    std::string file_path;
    std::unique_ptr<Parser> parser;
    bool reset{ false };

    bool is_tsv = false; // TODO: Remove when landmarks are dynamically computed

    void reset_data()
    {
        scale.clear();
        param_names.clear();
        scale_mean = scale_var = false;

        parse = false;
        parser = nullptr;
        is_tsv = false; // TODO: Remove when landmarks are dynamically computed
        reset = false;
    }
};

#endif // #ifndef UI_DATA_H
