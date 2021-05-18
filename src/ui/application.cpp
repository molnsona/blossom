#include "application.h"

#include <iostream>

// https://github.com/juliettef/IconFontCppHeaders
#include <IconsFontAwesome5.h>

using namespace Magnum;

Application::Application(const Arguments& arguments):
    Platform::Application{arguments, Configuration{}
        .setTitle("BlosSOM")
        .setWindowFlags(Configuration::WindowFlag::Resizable)}
{
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    
    init_imgui();
    
    /* Set up meshes */
    _plane = MeshTools::compile(Primitives::planeSolid(Primitives::PlaneFlag::TextureCoordinates));    
     /* Set up objects */
    (*(_objects[0] = new PickableObject{3, _textured_shader, _bg_color, _plane, _scene, _drawables}))
        .rotateX(-90.0_degf)        
        .scale(Vector3{20.0f, 0.0f, 20.0f});
    
    /* Configure camera */
    _cameraObject = new Object3D{&_scene};
    (*_cameraObject)
        .translate(Vector3::zAxis(20.0f))
       // .translate(Vector3::yAxis(5.0f))
        .rotateX(-90.0_degf);
    camera_trans.z() = 20.0f;

    _camera = new SceneGraph::Camera3D{*_cameraObject};
    Vector2 view_size(40.0f, 40.0f);
    _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
        .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
        .setViewport(GL::defaultFramebuffer.viewport().size());

    
    /* Loop at 60 Hz max */
    setSwapInterval(1);
    setMinimalLoopPeriod(16);
}

void Application::init_imgui() {
    
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

		font_texture.setMagnificationFilter(GL::SamplerFilter::Linear)
		    .setMinificationFilter(GL::SamplerFilter::Linear)
		    .setStorage(1, GL::TextureFormat::RGBA8, image.size())
		    .setSubImage(0, {}, image);

		io.Fonts->TexID = static_cast<void*>(&font_texture);

		io.FontDefault = _p_font;
	}

    ImFontConfig config;
	config.MergeMode = true;
	static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
	io.Fonts->AddFontFromFileTTF(BLOSSOM_DATA_DIR "/fa-solid-900.ttf", 16.0f, &config, icon_ranges);

    _imgui_vars._imgui = ImGuiIntegration::Context(*ImGui::GetCurrentContext(),
        Vector2{windowSize()}/dpiScaling(), windowSize(), framebufferSize());

    /* Setup proper blending to be used by ImGui */
    GL::Renderer::setBlendEquation(
        GL::Renderer::BlendEquation::Add, GL::Renderer::BlendEquation::Add);
    GL::Renderer::setBlendFunction(
        GL::Renderer::BlendFunction::SourceAlpha,
        GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    
    ImGui::GetStyle().WindowRounding = 10.0f;   
}

void Application::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color|GL::FramebufferClear::Depth);
   
    (*_objects[0]).setConfig(_cell_cnt, _mean, _std_dev);

    _camera->draw(_drawables);

    _imgui_vars._imgui.newFrame();

    /* Enable text input, if needed */
    if(ImGui::GetIO().WantTextInput && !isTextInputActive())
        startTextInput();
    else if(!ImGui::GetIO().WantTextInput && isTextInputActive())
        stopTextInput();
   
    draw_add_window(show_tools, windowSize());
    if(show_tools) draw_tools_window(show_tools, show_config, &_textured_shader, _cell_cnt, _mean, _std_dev);
    if(show_config) draw_config_window(show_config, _cell_cnt, _mean, _std_dev);

    /* Update application cursor */
    _imgui_vars._imgui.updateApplicationCursor(*this);

    /* Set appropriate states. If you only draw ImGui, it is sufficient to
       just enable blending and scissor test in the constructor. */
    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);

    _imgui_vars._imgui.drawFrame();
    
    /* Reset state. Only needed if you want to draw something else with
       different state after. */
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::Blending);


    swapBuffers();
    redraw();
}

void Application::viewportEvent(ViewportEvent& event) {
    GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});
    _camera->setViewport(event.windowSize());

    _imgui_vars._imgui.relayout(Vector2{event.windowSize()}/event.dpiScaling(),
        event.windowSize(), event.framebufferSize());
}

void Application::keyPressEvent(KeyEvent& event) {
    if(_imgui_vars._imgui.handleKeyPressEvent(event)) return;

    /* Movement */
    float speed = 0.5f;
    switch(event.key()) 
    {
        case KeyEvent::Key::Down:
            _cameraObject->translate(Vector3::zAxis(speed));
            camera_trans.z() += speed;
            break;
        case KeyEvent::Key::Up:
            _cameraObject->translate(Vector3::zAxis(-speed));
            camera_trans.z() -= speed;
            break;
        case KeyEvent::Key::Left:
            _cameraObject->translate(Vector3::xAxis(-speed));
            camera_trans.x() -= speed;
            break;
        case KeyEvent::Key::Right:
            _cameraObject->translate(Vector3::xAxis(speed));
            camera_trans.x() += speed;
            break;
        case KeyEvent::Key::Space: {
                Vector2 view_size(++zoom_depth, ++zoom_depth);
                _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
                    .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
                    .setViewport(GL::defaultFramebuffer.viewport().size());
                break;
            }
        case KeyEvent::Key::Esc: {
                Vector2 view_size(--zoom_depth, --zoom_depth);
                _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
                    .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
                    .setViewport(GL::defaultFramebuffer.viewport().size());
                break;
            }
        // Reset camera, TODO
        case KeyEvent::Key::Tab: {
                zoom_depth = 40.0f;
                Vector2 view_size(zoom_depth, zoom_depth);
                _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
                    .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
                    .setViewport(GL::defaultFramebuffer.viewport().size());
                (*_cameraObject)
                    .translate(-camera_trans)
                // .translate(Vector3::yAxis(5.0f))
                    .rotateX(-90.0_degf);
                camera_trans = Vector3(0.0f, 0.0f, 0.0f);
                break;
            }            
    }
    event.setAccepted();
}

void Application::keyReleaseEvent(KeyEvent& event) {
   if(_imgui_vars._imgui.handleKeyReleaseEvent(event)) return;
}

void Application::mousePressEvent(MouseEvent& event) {
    if(_imgui_vars._imgui.handleMousePressEvent(event)) {
        event.setAccepted(true);
        return;
    }

    _mousePressPosition = event.position();
}

void Application::mouseReleaseEvent(MouseEvent& event) {
   if(_imgui_vars._imgui.handleMouseReleaseEvent(event)) {
        event.setAccepted(true);
        return;
   }

    // TODO add change speed of draging
    float speed = 2.5f;
    auto delta = Magnum::Vector2d(event.position()) - Magnum::Vector2d(_mousePressPosition);
    if(delta != Vector2d(0, 0)) {
        auto norm_delta = delta.normalized();
        _cameraObject->translate(Vector3::xAxis(-float(speed * norm_delta.x())));
        _cameraObject->translate(Vector3::zAxis(-float(speed * norm_delta.y())));
        camera_trans.x() = -float(speed * norm_delta.x());
        camera_trans.z() = -float(speed * norm_delta.y());
    }
}

void Application::mouseMoveEvent(MouseMoveEvent& event) {
   if(_imgui_vars._imgui.handleMouseMoveEvent(event)) {
        event.setAccepted(true);
        return;
    }
}

void Application::mouseScrollEvent(MouseScrollEvent& event) {
    if(_imgui_vars._imgui.handleMouseScrollEvent(event)) {
        /* Prevent scrolling the page */
        event.setAccepted();
        return;
    }

    if(!event.offset().y()) return;

    Vector2 view_size(zoom_depth, zoom_depth);
    if(1.0f - (event.offset().y()) > 0) 
        view_size = Vector2(++zoom_depth, ++zoom_depth);
    else {
        if(--zoom_depth <= 0) ++zoom_depth;
        view_size = Vector2(zoom_depth, zoom_depth);
    }

    _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
    .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
    .setViewport(GL::defaultFramebuffer.viewport().size());
}

void Application::textInputEvent(TextInputEvent& event) {
    if(_imgui_vars._imgui.handleTextInputEvent(event)) {
        event.setAccepted(true);
        return;
    }
}

MAGNUM_APPLICATION_MAIN(Application)
