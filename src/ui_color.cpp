
#include "application.h"
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
        auto dim = app.state.data.names.size();

        ImGui::Text("Column:");
        for (size_t i = 0; i < dim; ++i) {
            if (ImGui::RadioButton(
                  app.state.data.names[i].data(), &app.state.colors.color, i))
                app.state.colors.touch_config();
        }

        ImGui::End();
    }
}
