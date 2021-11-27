
#include "application.h"
#include "extern/colormap/palettes.hpp"

#include <exception>

uiColorSettings::uiColorSettings()
  : show_window(false)
{}

void
uiColorSettings::render(Application &app, ImGuiWindowFlags window_flags)
{
    if (!show_window)
        return;

    if (ImGui::Begin("Color", &show_window, window_flags)) {
        ImGui::Text("Color palette:");
        if (ImGui::BeginCombo("##palettes",
                              app.state.colors.col_palette.c_str())) {
            bool ret = false;
            for (auto &[name, _] : colormap::palettes) {
                bool is_selected = name == app.state.colors.col_palette;
                if (ImGui::Selectable(name.c_str(), &is_selected)) {
                    app.state.colors.col_palette = name;
                    ret = true;
                }
            }

            if (ret)
                app.state.colors.touch_config();
            ImGui::EndCombo();
        }

        ImGui::Text("Reverse color palette:");
        if (ImGui::Checkbox("##reverse", &app.state.colors.reverse))
            app.state.colors.touch_config();

        ImGui::Text("Alpha:");
        if (ImGui::SliderFloat("##alpharender",
                               &app.state.colors.alpha,
                               0.0f,
                               1.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp))
            app.state.colors.touch_config();

        ImGui::Text("Column:");
        auto dim = app.state.data.names.size();
        if (!dim)
            ImGui::Text("No columns are present.");

        for (size_t i = 0; i < dim; ++i) {
            if (ImGui::RadioButton(
                  app.state.data.names[i].data(), &app.state.colors.color, i))
                app.state.colors.touch_config();
        }

        ImGui::End();
    }
}
