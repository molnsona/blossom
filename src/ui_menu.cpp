
#include "application.h"

#include <IconsFontAwesome5.h>

constexpr float WINDOW_PADDING = 100.0f;
constexpr float TOOLS_HEIGHT = 325.0f;
constexpr float WINDOW_WIDTH = 50.0f;

uiMenu::uiMenu()
  : show_menu(false)
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
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_AlwaysAutoResize;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);

    draw_menu_button(show_menu, app.view.fb_size);
    if (show_menu)
        draw_menu_window(app.view.fb_size, ui);

    loader.render(app);
    scaler.render(app, window_flags);
    training_set.render(app, window_flags);

    if (show_color)
        draw_color_window(ui);

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
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

        menu_entry(ICON_FA_WRENCH, "Scale data", scaler);
        menu_entry(ICON_FA_SLIDERS_H, "Training settings", training_set);

        if (ImGui::Button(ICON_FA_PALETTE, ImVec2(50.75f, 50.75f))) {
            show_color = true;
            show_menu = false;
        }
        tooltip("Color points");

        ImGui::End();
    }

    ImGui::PopStyleVar();
}

void
uiMenu::draw_color_window(UiData &ui)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_AlwaysAutoResize;
    // ImGuiWindowFlags_NoTitleBar;

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
}
