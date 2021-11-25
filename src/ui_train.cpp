
#include "application.h"
#include <exception>

uiTrainingSettings::uiTrainingSettings()
  : show_window(false)
{}

void
uiTrainingSettings::render(Application &app)
{
    if (!show_window)
        return;

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_AlwaysAutoResize;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);

    if (ImGui::Begin("Training settings", &show_window, window_flags)) {

        ImGui::Checkbox("som", &app.state.training_conf.som_landmark);
        ImGui::Checkbox("kmeans",
                        &app.state.training_conf.kmeans_landmark);
        ImGui::Checkbox("knn", &app.state.training_conf.knn_edges);
        ImGui::Checkbox("layout", &app.state.training_conf.graph_layout);

        ImGui::SliderFloat("alpha",
                           &app.state.training_conf.alpha,
                           0.001f,
                           2.0f,
                           "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);

        ImGui::SliderFloat("sigma",
                           &app.state.training_conf.sigma,
                           0.1f,
                           5.0f,
                           "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);
    }

    ImGui::End();

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}
