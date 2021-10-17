
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/ImageView.h>

// https://github.com/juliettef/IconFontCppHeaders
#include <IconsFontAwesome5.h>

#include "imgui_config.h"
#include "ui_imgui.h"

UiImgui::UiImgui(const Platform::Application *app)
  : show_menu(false)
  , show_scale(false)
  , show_color(false)
{
    ImGui::CreateContext();
    ImGui::StyleColorsLight();

    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->AddFontDefault();
    {
        // Regular size
        p_font = io.Fonts->AddFontFromFileTTF(
          BLOSSOM_DATA_DIR "/SourceSansPro-Regular.ttf", 16);

        int width, height;
        unsigned char *pixels = nullptr;
        int pixelSize;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height, &pixelSize);

        ImageView2D image{ GL::PixelFormat::RGBA,
                           GL::PixelType::UnsignedByte,
                           { width, height },
                           { pixels,
                             std::size_t(pixelSize * width * height) } };

        font_texture.setMagnificationFilter(GL::SamplerFilter::Linear)
          .setMinificationFilter(GL::SamplerFilter::Linear)
          .setStorage(1, GL::TextureFormat::RGBA8, image.size())
          .setSubImage(0, {}, image);

        io.Fonts->TexID = static_cast<void *>(&font_texture);

        io.FontDefault = p_font;
    }

    ImFontConfig config;
    config.MergeMode = true;
    static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
    io.Fonts->AddFontFromFileTTF(
      BLOSSOM_DATA_DIR "/fa-solid-900.ttf", 16.0f, &config, icon_ranges);

    context = ImGuiIntegration::Context(*ImGui::GetCurrentContext(),
                                        Vector2{ app->windowSize() },
                                        app->windowSize(),
                                        app->framebufferSize());

    /* Setup proper blending to be used by ImGui */
    GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
                                   GL::Renderer::BlendEquation::Add);
    GL::Renderer::setBlendFunction(
      GL::Renderer::BlendFunction::SourceAlpha,
      GL::Renderer::BlendFunction::OneMinusSourceAlpha);

    ImGui::GetStyle().WindowRounding = 10.0f;

    // Uncomment to change colors of ui
    // ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(0, 0, 0, 100));
    // ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(148, 210, 189, 100));
    // ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(0, 0, 0, 100));
    // ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(0, 0, 0, 100));

    open_file.SetTitle("Open file");
    open_file.SetTypeFilters({ ".fcs", ".tsv" });
}

void
UiImgui::draw_event(const View &view, UiData &ui, Platform::Application *app)
{
    context.newFrame();

    /* Enable text input, if needed */
    if (ImGui::GetIO().WantTextInput && !app->isTextInputActive())
        app->startTextInput();
    else if (!ImGui::GetIO().WantTextInput && app->isTextInputActive())
        app->stopTextInput();

    draw_add_window(view.fb_size);
    if (show_menu)
        draw_menu_window(view.fb_size, ui);
    if (show_scale)
        draw_scale_window(ui.trans_data);
    if (show_color)
        draw_color_window(ui);

    draw_open_file(ui.parser_data);

    /* Update application cursor */
    context.updateApplicationCursor(*app);

    /* Set appropriate states. If you only draw ImGui, it is sufficient to
       just enable blending and scissor test in the constructor. */
    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    GL::Renderer::setBlendFunction(
      GL::Renderer::BlendFunction::SourceAlpha,
      GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);

    context.drawFrame();

    /* Reset state. Only needed if you want to draw something else with
       different state after. */
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::Blending);
}

void
UiImgui::viewport_event(Platform::Application::ViewportEvent &event)
{
    context.relayout(Vector2{ event.windowSize() },
                     event.windowSize(),
                     event.framebufferSize());
}

bool
UiImgui::key_press_event(Platform::Application::KeyEvent &event)
{
    return context.handleKeyPressEvent(event);
}

bool
UiImgui::key_release_event(Platform::Application::KeyEvent &event)
{
    return context.handleKeyReleaseEvent(event);
}

bool
UiImgui::mouse_press_event(Platform::Application::MouseEvent &event)
{
    return context.handleMousePressEvent(event);
}

bool
UiImgui::mouse_release_event(Platform::Application::MouseEvent &event)
{
    return context.handleMouseReleaseEvent(event);
}

bool
UiImgui::mouse_move_event(Platform::Application::MouseMoveEvent &event)
{
    return context.handleMouseMoveEvent(event);
}

bool
UiImgui::mouse_scroll_event(Platform::Application::MouseScrollEvent &event)
{
    return context.handleMouseScrollEvent(event);
}

bool
UiImgui::text_input_event(Platform::Application::TextInputEvent &event)
{
    return context.handleTextInputEvent(event);
}

void
UiImgui::draw_add_window(const Vector2i &window_size)
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
            show_menu = show_menu ? false : true;
        }

        ImGui::End();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
UiImgui::draw_menu_window(const Vector2i &window_size, UiData &ui)
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

        if (ImGui::Button(ICON_FA_FOLDER_OPEN, ImVec2(50.75f, 50.75f))) {
            open_file.Open();
            show_menu = false;
        }
        hover_info("Open file");

        if (ImGui::Button(ICON_FA_SAVE, ImVec2(50.75f, 50.75f))) {
            show_menu = false;
        }
        hover_info("Save");

        ImGui::Separator();

        if (ImGui::Button(ICON_FA_SLIDERS_H, ImVec2(50.75f, 50.75f))) {
            show_scale = true;
            show_menu = false;
        }
        hover_info("Scale data");

        if (ImGui::Button(ICON_FA_PALETTE, ImVec2(50.75f, 50.75f))) {
            show_color = true;
            show_menu = false;
        }
        hover_info("Color points");

        if (ImGui::Button(ICON_FA_UNDO, ImVec2(50.75f, 50.75f))) {
            ui.reset = true;
            show_menu = false;
        }
        hover_info("Reset");

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
UiImgui::draw_scale_window(UiTransData &ui)
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
UiImgui::draw_color_window(UiData &ui)
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
        if(ui.trans_data.param_names.size() == 0)  ImGui::Text("No columns detected.");
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

void
UiImgui::draw_open_file(UiParserData &ui)
{
    open_file.Display();

    if (open_file.HasSelected()) {
        std::string file_path = open_file.GetSelected().string();

        ui.parse = true;
        ui.file_path = file_path;

        open_file.ClearSelected();
    }
}

void
UiImgui::hover_info(const std::string &text)
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2, 2));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8f);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(text.data());
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}
