
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

    if (!ImGui::Begin("Training settings", &show_window, window_flags))
        return;

    ImGui::Checkbox("som", &app.state.training_conf.som_landmark);
    ImGui::Checkbox("kmeans", &app.state.training_conf.kmeans_landmark);
    ImGui::Checkbox("knn", &app.state.training_conf.knn_edges);
    ImGui::Checkbox("layout", &app.state.training_conf.graph_layout);

    // TODO: show parameters according to active training method

    ImGui::SliderFloat("alpha", // TODO: logarithmicly
                       &app.state.training_conf.alpha,
                       0.001f,
                       2.0f,
                       "%.3f",
                       ImGuiSliderFlags_AlwaysClamp);

    ImGui::SliderFloat("sigma", // TODO: odmocninovo
                       &app.state.training_conf.sigma,
                       0.1f,
                       5.0f,
                       "%.3f",
                       ImGuiSliderFlags_AlwaysClamp);

    ImGui::SliderInt("iters", &app.state.training_conf.iters, 0, 200);

    ImGui::SliderInt("kns", &app.state.training_conf.kns, 0, 10);

    ImGui::SliderInt("topn", &app.state.training_conf.topn, 3, 32);

    ImGui::SliderFloat("boost", // TODO: exponentially
                       &app.state.training_conf.boost,
                       0.2f,
                       5.0f,
                       "%.3f",
                       ImGuiSliderFlags_AlwaysClamp);

    ImGui::SliderFloat("adjust",
                       &app.state.training_conf.adjust,
                       0.0f,
                       2.0f,
                       "%.3f",
                       ImGuiSliderFlags_AlwaysClamp);

    ImGui::End();
}
