
#include "application.h"

#include <algorithm>
#include <exception>

uiExporter::uiExporter()
  : show_window(false)
  , opener(ImGuiFileBrowserFlags_SelectDirectory |
           ImGuiFileBrowserFlags_CreateNewDir)
{
    opener.SetTitle("Select directory");
}

void
uiExporter::render(Application &app, ImGuiWindowFlags window_flags)
{
    if (!show_window)
        return;

    opener.Display();

    if (opener.HasSelected()) {
        try {
            exporter.export_data(app.state, opener.GetSelected().string());
        } catch (std::exception &e) {
            loading_error = e.what();
        }

        opener.ClearSelected();
    }

    if (!loading_error.empty()) {
        ImGui::Begin("Loading error", nullptr, 0);
        ImGui::Text(loading_error.c_str());
        if (ImGui::Button("OK"))
            loading_error = "";
        ImGui::End();
    }

    if (ImGui::Begin("Export##window", &show_window, window_flags)) {
        auto export_line = [&](const char *text, int type) {
            ImGui::Text(text);
            std::string name = "##export_checkbox" + std::to_string(type);
            ImGui::Checkbox(name.data(), &exporter.data_flags[type]);
            ImGui::SameLine();
            name = "file name##export" + std::to_string(type);
            ImGui::InputText(
              name.data(), exporter.file_names[type].data(), file_name_size);
        };

        export_line("Export hidim points:", Exporter::Types::POINTS_HD);

        export_line("Export hidim landmarks:", Exporter::Types::LAND_HD);

        export_line("Export 2D points:", Exporter::Types::POINTS_2D);

        export_line("Export 2D landmarks:", Exporter::Types::LAND_2D);

        ImGui::NewLine();

        if (ImGui::Checkbox("Export all", &exporter.all))
            if (exporter.all)
                std::fill(
                  exporter.data_flags.begin(), exporter.data_flags.end(), true);
            else
                std::fill(exporter.data_flags.begin(),
                          exporter.data_flags.end(),
                          false);

        ImGui::NewLine();

        if (ImGui::Button("Export##button"))
            opener.Open();

        ImGui::End();
    }
}
