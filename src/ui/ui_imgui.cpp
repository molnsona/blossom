
#include <Magnum/ImageView.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>

// https://github.com/juliettef/IconFontCppHeaders
#include <IconsFontAwesome5.h>

#include "ui_imgui.h"
#include "imgui_config.h"

UiImgui::UiImgui(const Vector2i& window_size,
    const Vector2& dpi_scaling,
    const Vector2i& frame_buffer_size) 
{
    ImGui::CreateContext();
    ImGui::StyleColorsLight();

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault();
	{
		// Regular size
		_p_font = io.Fonts->AddFontFromFileTTF(BLOSSOM_DATA_DIR "/SourceSansPro-Regular.ttf", 16);

		int width, height;
		unsigned char* pixels = nullptr;
		int pixelSize;
		io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height, &pixelSize);

		ImageView2D image{ GL::PixelFormat::RGBA,
			               GL::PixelType::UnsignedByte,
			               { width, height },
			               { pixels, std::size_t(pixelSize * width * height) } };

		_font_texture.setMagnificationFilter(GL::SamplerFilter::Linear)
		    .setMinificationFilter(GL::SamplerFilter::Linear)
		    .setStorage(1, GL::TextureFormat::RGBA8, image.size())
		    .setSubImage(0, {}, image);

		io.Fonts->TexID = static_cast<void*>(&_font_texture);

		io.FontDefault = _p_font;
	}

    ImFontConfig config;
	config.MergeMode = true;
	static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
	io.Fonts->AddFontFromFileTTF(BLOSSOM_DATA_DIR "/fa-solid-900.ttf", 16.0f, &config, icon_ranges);

    _context = ImGuiIntegration::Context(*ImGui::GetCurrentContext(),
        Vector2{window_size}/dpi_scaling, window_size, frame_buffer_size);

    /* Setup proper blending to be used by ImGui */
    GL::Renderer::setBlendEquation(
        GL::Renderer::BlendEquation::Add, GL::Renderer::BlendEquation::Add);
    GL::Renderer::setBlendFunction(
        GL::Renderer::BlendFunction::SourceAlpha,
        GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    
    ImGui::GetStyle().WindowRounding = 10.0f;   
}

void UiImgui::draw_event(Platform::Application* app)
{
    _context.newFrame();

    /* Enable text input, if needed */
    if(ImGui::GetIO().WantTextInput && !app->isTextInputActive())
        app->startTextInput();
    else if(!ImGui::GetIO().WantTextInput && app->isTextInputActive())
        app->stopTextInput();
   
    draw_add_window(app->windowSize());
    if(_show_tools) draw_tools_window();
    if(_show_config) draw_config_window();

    /* Update application cursor */
    _context.updateApplicationCursor(*app);

    /* Set appropriate states. If you only draw ImGui, it is sufficient to
       just enable blending and scissor test in the constructor. */
    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);

    _context.drawFrame();
    
    /* Reset state. Only needed if you want to draw something else with
       different state after. */
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::Blending);
}

void UiImgui::viewport_event(Platform::Application::ViewportEvent& event)
{
    _context.relayout(Vector2{event.windowSize()}/event.dpiScaling(),
        event.windowSize(), event.framebufferSize());
}

bool UiImgui::key_press_event(Platform::Application::KeyEvent& event)
{
    return _context.handleKeyPressEvent(event);
}

bool UiImgui::key_release_event(Platform::Application::KeyEvent& event)
{
    return _context.handleKeyReleaseEvent(event);
}

bool UiImgui::mouse_press_event(Platform::Application::MouseEvent& event)
{
    return _context.handleMousePressEvent(event);
}

bool UiImgui::mouse_release_event(Platform::Application::MouseEvent& event)
{
    return _context.handleMouseReleaseEvent(event);
}

bool UiImgui::mouse_move_event(Platform::Application::MouseMoveEvent& event)
{
    return _context.handleMouseMoveEvent(event);
}

bool UiImgui::mouse_scroll_event(Platform::Application::MouseScrollEvent& event)
{
    return _context.handleMouseScrollEvent(event);
}

bool UiImgui::text_input_event(Platform::Application::TextInputEvent& event)
{
    return _context.handleTextInputEvent(event);
}

void UiImgui::draw_add_window(const Vector2i& window_size)
{
    ImGuiWindowFlags window_flags = 
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 50.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));        
    
    if(ImGui::Begin("Plus", nullptr, window_flags))
    {
        ImGui::SetWindowPos(ImVec2(
            static_cast<float>(window_size.x() - WINDOW_PADDING),
            static_cast<float>(window_size.y() - WINDOW_PADDING)));
        ImGui::SetWindowSize(ImVec2(WINDOW_WIDTH, WINDOW_WIDTH));

        if(ImGui::Button(ICON_FA_PLUS, ImVec2(50.75f, 50.75f))) {
            _show_tools = true;
        }

        ImGui::End();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void UiImgui::draw_tools_window() 
{
    ImGuiWindowFlags window_flags = 
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize;

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(ImVec2(WINDOW_PADDING - (WINDOW_WIDTH / 2), center.y), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(WINDOW_WIDTH, TOOLS_HEIGHT));

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));        

    if(ImGui::Begin("Tools", &_show_tools, window_flags)) {
        if(ImGui::Button(ICON_FA_COGS, ImVec2(50.75f, 50.75f))) {
            _show_config = true;
        }
        if(ImGui::Button(ICON_FA_UNDO, ImVec2(50.75f, 50.75f))) {
            _cell_cnt = 10000;
            _mean = 0;
            _std_dev = 300;
        }
        if(ImGui::Button(ICON_FA_TIMES, ImVec2(50.75f, 50.75f))) {
            _show_tools = false;
        }

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
}

void UiImgui::draw_config_window() 
{
    ImGuiWindowFlags window_flags = 
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_AlwaysAutoResize; //|
        //ImGuiWindowFlags_NoTitleBar;        

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
//    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));        

    if(ImGui::Begin("##Config", &_show_config, window_flags)) {     
        ImGui::SetNextItemWidth(200.0f); 
        ImGui::SliderInt("Cell count", &_cell_cnt, 0, 100000);

        ImGui::SetNextItemWidth(200.0f); 
        ImGui::SliderInt("Mean", &_mean, -2000, 2000);
        
        ImGui::SetNextItemWidth(200.0f); 
        ImGui::SliderInt("Std deviation", &_std_dev, 0, 1000);

        ImGui::End();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
  //  ImGui::PopStyleVar();
}