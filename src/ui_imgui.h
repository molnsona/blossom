#ifndef UI_IMGUI_H
#define UI_IMGUI_H

#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Math/Color.h>
#include <Magnum/Platform/Sdl2Application.h>

#include "imfilebrowser.h"
#include "state.h"
#include "view.h"

using namespace Magnum;
using namespace Math::Literals;

class UiImgui
{
public:
    UiImgui() = delete;
    UiImgui(const Platform::Application *app);

    void draw_event(const View &view, State *state, Platform::Application *app);

    void viewport_event(Platform::Application::ViewportEvent &event);
    bool key_press_event(Platform::Application::KeyEvent &event);
    bool key_release_event(Platform::Application::KeyEvent &event);
    bool mouse_press_event(Platform::Application::MouseEvent &event);
    bool mouse_release_event(Platform::Application::MouseEvent &event);
    bool mouse_move_event(Platform::Application::MouseMoveEvent &event);
    bool mouse_scroll_event(Platform::Application::MouseScrollEvent &event);
    bool text_input_event(Platform::Application::TextInputEvent &event);

private:
    void draw_add_window(const Vector2i &window_size);
    void draw_menu_window(const Vector2i &window_size, State *p_state);
    void draw_config_window(State *p_state);
    void draw_open_file();

    ImGuiIntegration::Context context{ NoCreate };

    ImFont *p_font;
    GL::Texture2D font_texture;

    bool show_menu{ false };
    bool show_config{ false };

    ImGui::FileBrowser open_file;
};

#endif // #ifndef UI_IMGUI_H
