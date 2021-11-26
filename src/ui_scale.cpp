
#include "application.h"
#include <exception>

uiScaler::uiScaler()
  : show_window(false)
{}

void
uiScaler::render(Application &app, ImGuiWindowFlags window_flags)
{
    if (!show_window)
        return;

    // TODO this needs to be changed to the TransData substructures and
    // ScaledData configs

    if (app.state.data.names.size() != app.state.trans.dim()) {
        ImGui::Begin("Data error", nullptr, 0);
        ImGui::Text("Data has different dimension than transformed data.");
        if (ImGui::Button("OK"))
            return;
        ImGui::End();
    }

    auto dim = app.state.trans.dim();

    if (!ImGui::Begin("Scale", &show_window, window_flags))
        return;

    if (!ImGui::BeginTable("##table", 5))
        return;

    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    ImGui::Text("asinh");
    ImGui::TableNextColumn();
    ImGui::Text("asinh cofactor");
    ImGui::TableNextColumn();
    ImGui::Text("zscale");
    ImGui::TableNextColumn();
    ImGui::Text("affine adjust");

    for (size_t i = 0; i < dim; ++i) {
        ImGui::TableNextColumn();
        ImGui::Text(app.state.data.names[i].c_str());

        ImGui::TableNextColumn();
        std::string name = "##asinh" + std::to_string(i);
        if (ImGui::Checkbox(name.data(), &app.state.trans.config[i].asinh))
            app.state.trans.touch_config();

        ImGui::TableNextColumn();
        name = "##asinh_cofactor" + std::to_string(i);
        if (ImGui::SliderFloat(name.data(),
                               &app.state.trans.config[i].asinh_cofactor,
                               0.1f,
                               1000.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp))
            app.state.trans.touch_config();

        ImGui::TableNextColumn();
        name = "##zscale" + std::to_string(i);
        if (ImGui::Checkbox(name.data(), &app.state.trans.config[i].zscale))
            app.state.trans.touch_config();

        ImGui::TableNextColumn();
        name = "##affine_adjust" + std::to_string(i);
        if (ImGui::SliderFloat(name.data(),
                               &app.state.trans.config[i].affine_adjust,
                               -3.0f,
                               3.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp))
            app.state.trans.touch_config();
    }

    ImGui::EndTable();
    ImGui::End();
#if 0
    if (!ImGui::Begin("Scale", &show_window, window_flags)) return;

    // ImGui::Text("Scale:");
    app.state.trans_config.mean_changed |=
      ImGui::Checkbox("Mean (=0)", &app.state.trans_config.scale_mean);
    app.state.trans_config.var_changed |=
      ImGui::Checkbox("Variance (=1)", &app.state.trans_config.scale_var);
    app.state.trans_config.data_changed |=
      app.state.trans_config.mean_changed;
    app.state.trans_config.data_changed |=
      app.state.trans_config.var_changed;

    std::size_t i = 0;
    for (auto &&name : app.state.trans_config.param_names) {
        ImGui::SetNextItemWidth(200.0f);
        bool tmp = app.state.trans_config.sliders[i];
        tmp |= ImGui::SliderFloat(name.data(),
                      &app.state.trans_config.scales[i],
                      1.0f,
                      10.0f,
                      "%.3f",
                      ImGuiSliderFlags_AlwaysClamp);
        app.state.trans_config.sliders[i] = tmp;
        app.state.trans_config.sliders_changed |= tmp;
        app.state.trans_config.data_changed |=
          app.state.trans_config.sliders_changed;
        ++i;
    }

    ImGui::End();
#endif
}
