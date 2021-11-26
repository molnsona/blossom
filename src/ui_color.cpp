
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

    if (app.state.data.names.size() != app.state.colors.data.size()) {
        ImGui::Begin("Data error", nullptr, 0);
        ImGui::Text("Data has different dimension than colors data.");
        if (ImGui::Button("OK"))
            return;
        ImGui::End();
    }

    auto dim = app.state.data.names.size();

    if (!ImGui::Begin("Color", &show_window, window_flags))
        return;

    ImGui::Text("Column:");
    for (size_t i = 0; i < dim; ++i) {
        if (ImGui::RadioButton(
              app.state.data.names[i].data(), &app.state.colors.color, i))
            app.state.colors.touch_config();
    }

    ImGui::End();
}
