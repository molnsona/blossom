
#include "application.h"
#include <exception>

uiTrainingSettings::uiTrainingSettings()
  : show_window(false)
{}

void
uiTrainingSettings::render(Application &app, ImGuiWindowFlags window_flags)
{
    if (!show_window)
        return;

    if (ImGui::Begin("Training settings", &show_window, window_flags)) {

        ImGui::Checkbox("som", &app.state.training_conf.som_landmark);
        ImGui::Checkbox("kmeans", &app.state.training_conf.kmeans_landmark);
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
}
