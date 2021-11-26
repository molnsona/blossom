
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

    if (!ImGui::Begin("Color", &show_window, window_flags))
        return;

    ImGui::Text("Column:");
#if 0
if (app.state.trans_config.param_names.size() == 0)
    ImGui::Text("No columns detected.");
std::size_t i = 0;
for (auto &&name : app.state.trans_config.param_names) {
    ImGui::RadioButton(
      name.data(), &app.state.trans_config.color_ind, i);
    ++i;
}
#endif

    ImGui::End();
}
