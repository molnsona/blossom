
#include "application.h"

#include <IconsFontAwesome5.h>

constexpr float WINDOW_PADDING = 100.0f;
constexpr float TOOLS_HEIGHT = 325.0f;
constexpr float WINDOW_WIDTH = 50.0f;

uiMenu::uiMenu()
  : show_menu(false)
  , show_scale(false)
  , show_color(false)
{}

static void
draw_menu_button(bool &show_menu, const Vector2i &window_size)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoMove;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("Plus", nullptr, window_flags)) {
        ImGui::SetWindowPos(ImVec2(window_size.x() - WINDOW_PADDING,
                                   window_size.y() - WINDOW_PADDING));
        ImGui::SetWindowSize(ImVec2(WINDOW_WIDTH, WINDOW_WIDTH));

        if (ImGui::Button(ICON_FA_PLUS, ImVec2(50.75f, 50.75f))) {
            show_menu = !show_menu;
        }

        ImGui::End();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
uiMenu::render(Application &app)
{
    draw_menu_button(show_menu, app.view.fb_size);
    if (show_menu)
        draw_menu_window(app.view.fb_size, ui);

    loader.render(app);
    training_set.render(app);

    if (show_scale)
        draw_scale_window(ui.trans_data);
    if (show_color)
        draw_color_window(ui);
}

static void
tooltip(const char *text)
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2, 2));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8f);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(text);
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
uiMenu::draw_menu_window(const Vector2i &window_size, UiData &ui)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoMove;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("Tools", &show_menu, window_flags)) {
        ImGui::SetWindowPos(
          ImVec2(window_size.x() - WINDOW_PADDING,
                 window_size.y() - WINDOW_PADDING - TOOLS_HEIGHT));
        ImGui::SetWindowSize(ImVec2(WINDOW_WIDTH, TOOLS_HEIGHT));

        auto menu_entry = [&](auto icon, const char *label, auto &x) {
            if (ImGui::Button(icon, ImVec2(50.75f, 50.75f))) {
                x.show();
                show_menu = false;
            }
            tooltip(label);
        };

        menu_entry(ICON_FA_FOLDER_OPEN, "Open file", loader);

        // TODO convert these to menu_entries
        if (ImGui::Button(ICON_FA_SAVE, ImVec2(50.75f, 50.75f))) {
            show_menu = false;
        }
        tooltip("Save");

        ImGui::Separator();

        if (ImGui::Button(ICON_FA_WRENCH, ImVec2(50.75f, 50.75f))) {
            show_scale = true;
            show_menu = false;
        }
        tooltip("Scale data");

        menu_entry(ICON_FA_SLIDERS_H, "Training settings", training_set);

        if (ImGui::Button(ICON_FA_PALETTE, ImVec2(50.75f, 50.75f))) {
            show_color = true;
            show_menu = false;
        }
        tooltip("Color points");

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
uiMenu::draw_scale_window(UiTransData &ui)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_AlwaysAutoResize;
    // ImGuiWindowFlags_NoTitleBar;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    // ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("Scale", &show_scale, window_flags)) {

        // ImGui::Text("Scale:");
        ui.mean_changed |= ImGui::Checkbox("Mean (=0)", &ui.scale_mean);
        ui.var_changed |= ImGui::Checkbox("Variance (=1)", &ui.scale_var);
        ui.data_changed |= ui.mean_changed;
        ui.data_changed |= ui.var_changed;

        std::size_t i = 0;
        for (auto &&name : ui.param_names) {
            ImGui::SetNextItemWidth(200.0f);
            bool tmp = ui.sliders[i];
            tmp |= ImGui::SliderFloat(name.data(),
                                      &ui.scale[i],
                                      1.0f,
                                      10.0f,
                                      "%.3f",
                                      ImGuiSliderFlags_AlwaysClamp);
            ui.sliders[i] = tmp;
            ui.sliders_changed |= tmp;
            ui.data_changed |= ui.sliders_changed;
            ++i;
        }

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    //  ImGui::PopStyleVar();
}

void
uiMenu::draw_color_window(UiData &ui)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_AlwaysAutoResize;
    // ImGuiWindowFlags_NoTitleBar;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    // ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("Color", &show_color, window_flags)) {

        ImGui::Text("Column:");
        if (ui.trans_data.param_names.size() == 0)
            ImGui::Text("No columns detected.");
        std::size_t i = 0;
        for (auto &&name : ui.trans_data.param_names) {
            ImGui::RadioButton(name.data(), &ui.color_ind, i);
            ++i;
        }

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    //  ImGui::PopStyleVar();
}
