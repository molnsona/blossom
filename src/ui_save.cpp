
#include "application.h"

#include <algorithm>
#include <exception>
#include <fstream>

uiSaver::uiSaver()
  : show_window(false)
  , opener(ImGuiFileBrowserFlags_SelectDirectory |
           ImGuiFileBrowserFlags_CreateNewDir)
  , all(false)
  , data_flags{ false, false, false, false }
  , file_names{ "points_hd.tsv",
                "landmarks_hd.tsv",
                "points_2d.tsv",
                "landmarks_2d.tsv" }
{
    opener.SetTitle("Select directory");
}

void
uiSaver::render(Application &app, ImGuiWindowFlags window_flags)
{
    if (!show_window)
        return;

    opener.Display();

    if (opener.HasSelected()) {
        try {
            save_data(app.state, opener.GetSelected().string());
        } catch (std::exception &e) {
            saving_error = e.what();
        }

        opener.ClearSelected();
    }

    if (!saving_error.empty()) {
        ImGui::Begin("Saving error", nullptr, 0);
        ImGui::Text(saving_error.c_str());
        if (ImGui::Button("OK"))
            saving_error = "";
        ImGui::End();
    }

    if (ImGui::Begin("Save##window", &show_window, window_flags)) {
        auto save_line = [&](const char *text, int type) {
            ImGui::Text(text);
            std::string name = "##save_checkbox" + std::to_string(type);
            ImGui::Checkbox(name.data(), &data_flags[type]);
            ImGui::SameLine();
            name = "file name##save" + std::to_string(type);
            ImGui::InputText(
              name.data(), file_names[type].data(), file_name_size);
        };

        save_line("Save hidim points:", uiSaver::Types::POINTS_HD);

        save_line("Save hidim landmarks:", uiSaver::Types::LAND_HD);

        save_line("Save 2D points:", uiSaver::Types::POINTS_2D);

        save_line("Save 2D landmarks:", uiSaver::Types::LAND_2D);

        ImGui::NewLine();

        if (ImGui::Checkbox("Save all", &all))
            if (all)
                std::fill(data_flags.begin(), data_flags.end(), true);
            else
                std::fill(data_flags.begin(), data_flags.end(), false);

        ImGui::NewLine();

        if (ImGui::Button("Save##button"))
            opener.Open();

        ImGui::End();
    }
}

void
uiSaver::save_data(const State &state, const std::string &dir_name)
{
    if (data_flags[uiSaver::Types::POINTS_HD])
        write(uiSaver::Types::POINTS_HD, state, dir_name);

    if (data_flags[uiSaver::Types::LAND_HD])
        write(uiSaver::Types::LAND_HD, state, dir_name);

    if (data_flags[uiSaver::Types::POINTS_2D])
        write(uiSaver::Types::POINTS_2D, state, dir_name);

    if (data_flags[uiSaver::Types::LAND_2D])
        write(uiSaver::Types::LAND_2D, state, dir_name);
}

static void
write_data_float(size_t size,
                 size_t dim,
                 const std::vector<float> &data,
                 std::ofstream &handle)
{
    for (size_t i = 0; i < size; i += dim) {
        for (size_t j = 0; j < dim - 1; ++j) {
            handle << data[i + j] << '\t';
        }
        handle << data[i + dim - 1] << '\n';
    }
};

static void
write_data_2d(size_t size,
              const std::vector<Magnum::Vector2> &data,
              std::ofstream &handle)
{
    for (size_t i = 0; i < size; ++i) {
        handle << data[i].x() << '\t' << data[i].y() << '\n';
    }
};

void
uiSaver::write(Types type, const State &state, const std::string &dir_name)
{
    std::string path = dir_name + "/" + file_names[type];
    std::ofstream handle(path, std::ios::out);
    if (!handle)
        throw std::domain_error("Can not open file");

    switch (type) {
        case uiSaver::Types::POINTS_HD:
            write_data_float(
              state.data.data.size(), state.data.d, state.data.data, handle);

            break;
        case uiSaver::Types::LAND_HD:
            write_data_float(state.landmarks.hidim_vertices.size(),
                             state.landmarks.d,
                             state.landmarks.hidim_vertices,
                             handle);
            break;
        case uiSaver::Types::POINTS_2D:
            write_data_2d(
              state.scatter.points.size(), state.scatter.points, handle);
            break;
        case uiSaver::Types::LAND_2D:
            write_data_2d(state.landmarks.lodim_vertices.size(),
                          state.landmarks.lodim_vertices,
                          handle);
            break;
    }

    handle.close();
}
