#include "wrapper_imgui.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "vendor/IconsFontAwesome5.h"

bool
ImGuiWrapper::init(GLFWwindow *window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    if (!ImGui_ImplGlfw_InitForOpenGL(window, true))
        return false;
    if (!ImGui_ImplOpenGL3_Init("#version 330 core"))
        return false;
    ImGui::StyleColorsLight();

    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF(BLOSSOM_DATA_DIR "/SourceSansPro-Regular.ttf",
                                 16);

    ImFontConfig config;
    config.MergeMode = true;
    static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
    io.Fonts->AddFontFromFileTTF(
      BLOSSOM_DATA_DIR "/fa-solid-900.ttf", 16.0f, &config, icon_ranges);

    ImGui::GetStyle().WindowRounding = 10.0f;

    return true;
}

void
ImGuiWrapper::render(InputData &input, State &state)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    menu.render(input.fb_width, input.fb_height, state);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void
ImGuiWrapper::destroy()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
