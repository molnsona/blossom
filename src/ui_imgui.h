#ifndef UI_IMGUI_H
#define UI_IMGUI_H

#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Platform/Sdl2Application.h>

#include "state.h"

using namespace Magnum;
using namespace Math::Literals;

class UiImgui {
public:
    UiImgui() = delete;
    UiImgui(const Platform::Application* app);

    void draw_event(State* state, Platform::Application* app);
    
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
    void draw_tools_window(State* p_state);
    void draw_config_window(State* p_state);

    ImGuiIntegration::Context _context{NoCreate};

    ImFont* _p_font;
	GL::Texture2D _font_texture;

    bool _show_tools = false;
    bool _show_config = false;
};

#endif // #ifndef UI_IMGUI_H
