
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
                              app.state.colors.col_pallette.c_str())) {
            bool ret = false;
            for (auto &[name, _] : colormap::palettes) {
                bool is_selected = name == app.state.colors.col_pallette;
                if (ImGui::Selectable(name.c_str(), &is_selected)) {
                    app.state.colors.col_pallette = name;
                    ret = true;
                }
            }

            if (ret)
                app.state.colors.touch_config();
            ImGui::EndCombo();
        }

        auto dim = app.state.data.names.size();

        ImGui::Text("Column:");

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
