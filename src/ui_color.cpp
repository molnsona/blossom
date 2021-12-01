
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
        ImGui::Text("Coloring method:");
        if (ImGui::RadioButton("Expression coloring",
                               &app.state.colors.coloring,
                               (int)ColorData::Coloring::EXPR))
            app.state.colors.touch_config();

        if (ImGui::RadioButton("Cluster coloring",
                               &app.state.colors.coloring,
                               (int)ColorData::Coloring::CLUSTER))
            app.state.colors.touch_config();

        if (ImGui::SliderFloat("Alpha##color",
                               &app.state.colors.alpha,
                               0.0f,
                               1.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp))
            app.state.colors.touch_config();

        ImGui::Separator();

        switch (app.state.colors.coloring) {
            case (int)ColorData::Coloring::EXPR: {
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

                if (ImGui::Checkbox("Reverse color palette",
                                    &app.state.colors.reverse))
                    app.state.colors.touch_config();

                ImGui::Text("Column:");
                auto dim = app.state.data.names.size();
                if (!dim)
                    ImGui::Text("No columns are present.");

                for (size_t i = 0; i < dim; ++i) {
                    if (ImGui::RadioButton(app.state.data.names[i].data(),
                                           &app.state.colors.color,
                                           i))
                        app.state.colors.touch_config();
                }
            } break;
            case int(ColorData::Coloring::CLUSTER):
                if (ImGui::SliderInt(
                      "Cluster count", &app.state.colors.cluster_cnt, 1, 50))
                    app.state.colors.touch_config();
                break;
        }

        ImGui::End();
    }
}
