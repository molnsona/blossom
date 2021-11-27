
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

    if (app.state.data.names.size() != app.state.trans.dim()) {
        ImGui::Begin("Data error", nullptr, window_flags);
        ImGui::Text("Data has different dimension than transformed data.");
        if (ImGui::Button("OK")) {
            show_window = false;
        }
        ImGui::End();
        return;
    }

    if (app.state.data.names.size() != app.state.scaled.dim()) {
        ImGui::Begin("Data error", nullptr, window_flags);
        ImGui::Text("Data has different dimension than scaled data.");
        if (ImGui::Button("OK")) {
            show_window = false;
        }
        ImGui::End();
        return;
    }

    auto dim = app.state.trans.dim();

    if (ImGui::Begin("Scale", &show_window, window_flags)) {

        ImGui::BeginTable("##tabletrans", 7);

        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        ImGui::Text("asinh");
        ImGui::TableNextColumn();
        ImGui::Text("asinh cofactor");
        ImGui::TableNextColumn();
        ImGui::Text("zscale");
        ImGui::TableNextColumn();
        ImGui::Text("affine adjust");
        ImGui::TableNextColumn();
        ImGui::Text("scale");
        ImGui::TableNextColumn();
        ImGui::Text("sdev");

        for (size_t i = 0; i < dim; ++i) {
            ImGui::TableNextColumn();
            ImGui::Text(app.state.data.names[i].c_str());

            ImGui::TableNextColumn();
            std::string name = "##asinh" + std::to_string(i);
            if (ImGui::Checkbox(name.data(), &app.state.trans.config[i].asinh))
                app.state.trans.touch_config();

            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(slider_width);
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
            ImGui::SetNextItemWidth(slider_width);
            name = "##affine_adjust" + std::to_string(i);
            if (ImGui::SliderFloat(name.data(),
                                   &app.state.trans.config[i].affine_adjust,
                                   -3.0f,
                                   3.0f,
                                   "%.3f",
                                   ImGuiSliderFlags_AlwaysClamp))
                app.state.trans.touch_config();
        
            ImGui::TableNextColumn();
            name = "##scale" + std::to_string(i);
            if (ImGui::Checkbox(name.data(), &app.state.scaled.config[i].scale))
                app.state.scaled.touch_config();

            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(slider_width);
            name = "##sdev" + std::to_string(i);
            if (ImGui::SliderFloat(name.data(),
                                   &app.state.scaled.config[i].sdev,
                                   0.1f,
                                   5.0f,
                                   "%.3f",
                                   ImGuiSliderFlags_AlwaysClamp))
                app.state.scaled.touch_config();
        }

        ImGui::EndTable();
        ImGui::End();
    }
}
