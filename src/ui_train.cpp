
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
        if (ImGui::CollapsingHeader("SOM")) {
            ImGui::Checkbox("SOM##checkbox",
                            &app.state.training_conf.som_landmark);
            ImGui::SliderInt(
              "Iterations##SOM", &app.state.training_conf.som_iters, 0, 200);

            ImGui::SliderFloat("Alpha##SOM", // TODO: logarithmically
                               &app.state.training_conf.som_alpha,
                               0.001f,
                               2.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp);
            ImGui::SliderFloat("Sigma", // TODO: odmocninovo
                               &app.state.training_conf.sigma,
                               0.1f,
                               5.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp);
        }
        if (ImGui::CollapsingHeader("k-means")) {
            ImGui::Checkbox("k-means##checkbox",
                            &app.state.training_conf.kmeans_landmark);
            ImGui::SliderInt("Iterations##k-means",
                             &app.state.training_conf.kmeans_iters,
                             0,
                             200);
            ImGui::SliderFloat("Alpha##k-means", // TODO: logarithmically
                               &app.state.training_conf.kmeans_alpha,
                               0.001f,
                               2.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp);
            ImGui::SliderFloat("Gravity", // TODO: exponencialne
                               &app.state.training_conf.gravity,
                               0.0f,
                               0.1f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp);
        }
        if (ImGui::CollapsingHeader("k-NN graph")) {
            ImGui::Checkbox("Generate k-NN graph",
                            &app.state.training_conf.knn_edges);
            ImGui::SliderInt(
              "Graph neighbors (k)", &app.state.training_conf.kns, 0, 10);
        }

        if (ImGui::CollapsingHeader("Graph layout")) {
            ImGui::Checkbox("Layout the graph along the edge forces",
                            &app.state.training_conf.graph_layout);
        }

        if (ImGui::CollapsingHeader("t-SNE")) {
            ImGui::Checkbox("Layout the landmarks with t-SNE",
                            &app.state.training_conf.tsne_layout);
            ImGui::SliderInt("k neighbors",
                             &app.state.training_conf.tsne_k,
                             3,
                             app.state.landmarks.n_landmarks());
        }

        if (ImGui::CollapsingHeader("EmbedSOM")) {
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
        }

        ImGui::End();
    }
}
