
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
#if 0
    if (ImGui::Begin("Scale", &show_window, window_flags)) {
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
    }
#endif
}
