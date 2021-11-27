
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

        ImGui::SliderInt("Iterations (SOM and k-means)",
                         &app.state.training_conf.iters,
                         0,
                         200);
        ImGui::SliderFloat("Alpha (SOM and k-means)", // TODO: logarithmicly
                           &app.state.training_conf.alpha,
                           0.001f,
                           2.0f,
                           "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);

        ImGui::Checkbox("SOM", &app.state.training_conf.som_landmark);
        ImGui::SliderFloat("Sigma", // TODO: odmocninovo
                           &app.state.training_conf.sigma,
                           0.1f,
                           5.0f,
                           "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);

        ImGui::Checkbox("k-means", &app.state.training_conf.kmeans_landmark);
        ImGui::SliderFloat("Gravity", // TODO: exponencialne
                           &app.state.training_conf.gravity,
                           0.0f,
                           0.1f,
                           "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);

        ImGui::Checkbox("Generate k-NN graph",
                        &app.state.training_conf.knn_edges);
        ImGui::SliderInt(
          "Graph neighbors (k)", &app.state.training_conf.kns, 0, 10);
        ImGui::Checkbox("Layout the graph along the edge forces",
                        &app.state.training_conf.graph_layout);
        ImGui::Checkbox("Layout the landmarks with t-SNE",
                        &app.state.training_conf.tsne_layout);

        // TODO: show parameters according to active training method

        ImGui::Text("EmbedSOM settings");

        ImGui::SliderInt("k (landmark neighborhood size)",
                         &app.state.training_conf.topn,
                         3,
                         32);

        ImGui::SliderFloat("Boost", // TODO: use smooth instead
                           &app.state.training_conf.boost,
                           0.2f,
                           5.0f,
                           "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);

        ImGui::SliderFloat("Adjust",
                           &app.state.training_conf.adjust,
                           0.0f,
                           2.0f,
                           "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);

        ImGui::End();
    }
}
