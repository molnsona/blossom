
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/ImageView.h>

#include "vendor/IconsFontAwesome5.h"

#include "application.h"

UiImgui::UiImgui(const Application &app)
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
                                        Vector2{ app.windowSize() },
                                        app.windowSize(),
                                        app.framebufferSize());

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
}

void
UiImgui::draw_event(Application &app)
{
    context.newFrame();

    /* Enable text input, if needed */
    if (ImGui::GetIO().WantTextInput && !app.isTextInputActive())
        app.startTextInput();
    else if (!ImGui::GetIO().WantTextInput && app.isTextInputActive())
        app.stopTextInput();

    menu.render(app);

    /* Update application cursor */
    context.updateApplicationCursor(app);

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
