#ifndef UI_IMGUI_H
#define UI_IMGUI_H

#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Platform/Sdl2Application.h>

using namespace Magnum;
using namespace Math::Literals;

//class Application;

class UiImgui {
public:
    UiImgui(const Vector2i& window_size,
        const Vector2& dpi_scaling,
        const Vector2i& frame_buffer_size);

    void draw_event(Platform::Application* app);
    
    void viewport_event(Platform::Application::ViewportEvent& event);
    bool key_press_event(Platform::Application::KeyEvent& event);
    bool key_release_event(Platform::Application::KeyEvent& event);
    bool mouse_press_event(Platform::Application::MouseEvent& event);
    bool mouse_release_event(Platform::Application::MouseEvent& event);
    bool mouse_move_event(Platform::Application::MouseMoveEvent& event);
    bool mouse_scroll_event(Platform::Application::MouseScrollEvent& event);
    bool text_input_event(Platform::Application::TextInputEvent& event);
private:
    void draw_add_window(const Vector2i& window_size);
    void draw_tools_window();
    void draw_config_window();

    ImGuiIntegration::Context _context{NoCreate};

    Color4 _clearColor = 0x72909aff_rgbaf;
    Float _floatValue = 0.0f;

    ImFont* _p_font;
	GL::Texture2D _font_texture;

    bool _show_tools = false;
    bool _show_config = false;

    int _cell_cnt = 10000;
    int _mean = 0;
    int _std_dev = 300;

};

#endif // #ifndef UI_IMGUI_H