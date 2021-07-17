#ifndef IMGUI_UTILS_HPP
#define IMGUI_UTILS_HPP

// https://github.com/juliettef/IconFontCppHeaders
#include <IconsFontAwesome5.h>

#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Platform/Sdl2Application.h>

#include "imgui_config.h"

using namespace Magnum;
using namespace Math::Literals;

void
draw_add_window(bool &show_tools, const Vector2i &window_size)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoMove;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("Plus", nullptr, window_flags)) {
        ImGui::SetWindowPos(
          ImVec2(static_cast<float>(window_size.x() - WINDOW_PADDING),
                 static_cast<float>(window_size.y() - WINDOW_PADDING)));
        ImGui::SetWindowSize(ImVec2(WINDOW_WIDTH, WINDOW_WIDTH));

        if (ImGui::Button(ICON_FA_PLUS, ImVec2(50.75f, 50.75f))) {
            show_tools = true;
        }

        ImGui::End();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
draw_tools_window(bool &show_tools,
                  bool &show_config,
                  int &cell_cnt,
                  int &mean,
                  int &std_dev)
{
    ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize;

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(
      ImVec2(WINDOW_PADDING - (WINDOW_WIDTH / 2), center.y),
      ImGuiCond_Appearing,
      ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(WINDOW_WIDTH, TOOLS_HEIGHT));

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("Tools", &show_tools, window_flags)) {
        if (ImGui::Button(ICON_FA_COGS, ImVec2(50.75f, 50.75f))) {
            show_config = true;
        }
        if (ImGui::Button(ICON_FA_UNDO, ImVec2(50.75f, 50.75f))) {
            cell_cnt = 10000;
            mean = 0;
            std_dev = 300;
        }
        if (ImGui::Button(ICON_FA_TIMES, ImVec2(50.75f, 50.75f))) {
            show_tools = false;
        }

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
draw_config_window(bool &show_config, int &cell_cnt, int &mean, int &std_dev)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_AlwaysAutoResize; //|
    // ImGuiWindowFlags_NoTitleBar;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    //    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("##Config", &show_config, window_flags)) {
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderInt("Cell count", &cell_cnt, 0, 100000);

        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderInt("Mean", &mean, -2000, 2000);

        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderInt("Std deviation", &std_dev, 0, 1000);

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    //  ImGui::PopStyleVar();
}

#endif //#ifndef IMGUI_UTILS_HPP