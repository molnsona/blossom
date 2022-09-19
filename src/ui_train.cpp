/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
 *                    Sona Molnarova
 *
 * BlosSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * BlosSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * BlosSOM. If not, see <https://www.gnu.org/licenses/>.
 */

#include "ui_train.h"

#include "utils_imgui.hpp"

UiTrainingSettings::UiTrainingSettings()
  : show_window(false)
{}

void
UiTrainingSettings::render(State& state, ImGuiWindowFlags window_flags)
{
    if (!show_window)
        return;

    if (ImGui::Begin("Training settings", &show_window, window_flags)) {
        if (reset_button()) {
            state.training_conf.reset_data();
        }

        if (ImGui::CollapsingHeader("SOM")) {
            ImGui::Checkbox("SOM##checkbox",
                            &state.training_conf.som_landmark);
            ImGui::SliderInt(
              "Iterations##SOM", &state.training_conf.som_iters, 0, 200);

            ImGui::SliderFloat("Alpha##SOM", // TODO: logarithmically
                               &state.training_conf.som_alpha,
                               0.001f,
                               2.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp);
            ImGui::SliderFloat("Sigma", // TODO: odmocninovo
                               &state.training_conf.sigma,
                               0.1f,
                               5.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp);
        }
        if (ImGui::CollapsingHeader("k-means")) {
            ImGui::Checkbox("k-means##checkbox",
                            &state.training_conf.kmeans_landmark);
            ImGui::SliderInt("Iterations##k-means",
                             &state.training_conf.kmeans_iters,
                             0,
                             200);
            ImGui::SliderFloat("Alpha##k-means", // TODO: logarithmically
                               &state.training_conf.kmeans_alpha,
                               0.001f,
                               2.0f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp);
            ImGui::SliderFloat("Gravity", // TODO: exponencialne
                               &state.training_conf.gravity,
                               0.0f,
                               0.1f,
                               "%.3f",
                               ImGuiSliderFlags_AlwaysClamp);
        }
        if (ImGui::CollapsingHeader("k-NN graph")) {
            ImGui::Checkbox("Generate k-NN graph",
                            &state.training_conf.knn_edges);
            ImGui::SliderInt(
              "Graph neighbors (k)", &state.training_conf.kns, 0, 10);
        }

        if (ImGui::CollapsingHeader("Graph layout")) {
            ImGui::Checkbox("Layout the graph along the edge forces",
                            &state.training_conf.graph_layout);
        }

        if (ImGui::CollapsingHeader("t-SNE")) {
            ImGui::Checkbox("Layout the landmarks with t-SNE",
                            &state.training_conf.tsne_layout);
            ImGui::SliderInt("k neighbors",
                             &state.training_conf.tsne_k,
                             3,
                             state.landmarks.n_landmarks());
        }

        if (ImGui::CollapsingHeader("EmbedSOM")) {
            if (ImGui::SliderInt("k (landmark neighborhood size)",
                                 &state.training_conf.topn,
                                 3,
                                 32))
                state.scatter.touch_config();

            if (ImGui::SliderFloat("Boost", // TODO: use smooth instead
                                   &state.training_conf.boost,
                                   0.2f,
                                   5.0f,
                                   "%.3f",
                                   ImGuiSliderFlags_AlwaysClamp))
                state.scatter.touch_config();

            if (ImGui::SliderFloat("Adjust",
                                   &state.training_conf.adjust,
                                   0.0f,
                                   2.0f,
                                   "%.3f",
                                   ImGuiSliderFlags_AlwaysClamp))
                state.scatter.touch_config();
        }

        ImGui::End();
    }
}
