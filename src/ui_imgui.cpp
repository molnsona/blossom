
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/ImageView.h>

// https://github.com/juliettef/IconFontCppHeaders
#include <IconsFontAwesome5.h>

#include "imgui_config.h"
#include "ui_imgui.h"

UiImgui::UiImgui(const Platform::Application *app)
{
    ImGui::CreateContext();
    ImGui::StyleColorsLight();

    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->AddFontDefault();
    {
        // Regular size
        _p_font = io.Fonts->AddFontFromFileTTF(
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

        _font_texture.setMagnificationFilter(GL::SamplerFilter::Linear)
          .setMinificationFilter(GL::SamplerFilter::Linear)
          .setStorage(1, GL::TextureFormat::RGBA8, image.size())
          .setSubImage(0, {}, image);

        io.Fonts->TexID = static_cast<void *>(&_font_texture);

        io.FontDefault = _p_font;
    }

    ImFontConfig config;
    config.MergeMode = true;
    static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
    io.Fonts->AddFontFromFileTTF(
      BLOSSOM_DATA_DIR "/fa-solid-900.ttf", 16.0f, &config, icon_ranges);

    _context = ImGuiIntegration::Context(*ImGui::GetCurrentContext(),
                                         Vector2{ app->windowSize() } /
                                           app->dpiScaling(),
                                         app->windowSize(),
                                         app->framebufferSize());

    /* Setup proper blending to be used by ImGui */
    GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
                                   GL::Renderer::BlendEquation::Add);
    GL::Renderer::setBlendFunction(
      GL::Renderer::BlendFunction::SourceAlpha,
      GL::Renderer::BlendFunction::OneMinusSourceAlpha);

    ImGui::GetStyle().WindowRounding = 10.0f;
}

void
UiImgui::draw_event(State *p_state, Platform::Application *app)
{
    _context.newFrame();

    /* Enable text input, if needed */
    if (ImGui::GetIO().WantTextInput && !app->isTextInputActive())
        app->startTextInput();
    else if (!ImGui::GetIO().WantTextInput && app->isTextInputActive())
        app->stopTextInput();

    draw_add_window(app->windowSize());
    if (_show_tools)
        draw_tools_window(p_state);
    if (_show_config)
        draw_config_window(p_state);

    /* Update application cursor */
    _context.updateApplicationCursor(*app);

    /* Set appropriate states. If you only draw ImGui, it is sufficient to
       just enable blending and scissor test in the constructor. */
    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);

    _context.drawFrame();

    /* Reset state. Only needed if you want to draw something else with
       different state after. */
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::Blending);
}

void
UiImgui::viewport_event(Platform::Application::ViewportEvent &event)
{
    _context.relayout(Vector2{ event.windowSize() } / event.dpiScaling(),
                      event.windowSize(),
                      event.framebufferSize());
}

bool
UiImgui::key_press_event(Platform::Application::KeyEvent &event)
{
    return _context.handleKeyPressEvent(event);
}

bool
UiImgui::key_release_event(Platform::Application::KeyEvent &event)
{
    return _context.handleKeyReleaseEvent(event);
}

bool
UiImgui::mouse_press_event(Platform::Application::MouseEvent &event)
{
    return _context.handleMousePressEvent(event);
}

bool
UiImgui::mouse_release_event(Platform::Application::MouseEvent &event)
{
    return _context.handleMouseReleaseEvent(event);
}

bool
UiImgui::mouse_move_event(Platform::Application::MouseMoveEvent &event)
{
    return _context.handleMouseMoveEvent(event);
}

bool
UiImgui::mouse_scroll_event(Platform::Application::MouseScrollEvent &event)
{
    return _context.handleMouseScrollEvent(event);
}

bool
UiImgui::text_input_event(Platform::Application::TextInputEvent &event)
{
    return _context.handleTextInputEvent(event);
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
        ImGui::SetWindowPos(
          ImVec2(static_cast<float>(window_size.x() - WINDOW_PADDING),
                 static_cast<float>(window_size.y() - WINDOW_PADDING)));
        ImGui::SetWindowSize(ImVec2(WINDOW_WIDTH, WINDOW_WIDTH));

        if (ImGui::Button(ICON_FA_PLUS, ImVec2(50.75f, 50.75f))) {
            _show_tools = true;
        }

        ImGui::End();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
UiImgui::draw_tools_window(State *p_state)
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

    if (ImGui::Begin("Tools", &_show_tools, window_flags)) {
        if (ImGui::Button(ICON_FA_COGS, ImVec2(50.75f, 50.75f))) {
            _show_config = true;
        }
        if (ImGui::Button(ICON_FA_UNDO, ImVec2(50.75f, 50.75f))) {
            p_state->cell_cnt = 10000;
            p_state->mean = 0;
            p_state->std_dev = 300;
        }
        if (ImGui::Button(ICON_FA_TIMES, ImVec2(50.75f, 50.75f))) {
            _show_tools = false;
        }

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void
UiImgui::draw_config_window(State *p_state)
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_AlwaysAutoResize;
    // ImGuiWindowFlags_NoTitleBar;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    // ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (ImGui::Begin("##Config", &_show_config, window_flags)) {
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderInt("Cell count", &p_state->cell_cnt, 0, 100000);

        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderInt("Mean", &p_state->mean, -2000, 2000);

        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderInt("Std deviation", &p_state->std_dev, 0, 1000);

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    //  ImGui::PopStyleVar();
}
