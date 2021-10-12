#ifndef UI_IMGUI_H
#define UI_IMGUI_H

#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Math/Color.h>
#include <Magnum/Platform/Sdl2Application.h>

#include <string>

#include "extern/imfilebrowser.h"
#include "ui_data.h"
#include "ui_parser_data.h"
#include "ui_trans_data.h"
#include "view.h"

using namespace Magnum;
using namespace Math::Literals;

class UiImgui
{
public:
    UiImgui() = delete;
    UiImgui(const Platform::Application *app);

    void draw_event(const View &view, UiData &ui, Platform::Application *app);

    void viewport_event(Platform::Application::ViewportEvent &event);
    bool key_press_event(Platform::Application::KeyEvent &event);
    bool key_release_event(Platform::Application::KeyEvent &event);
    bool mouse_press_event(Platform::Application::MouseEvent &event);
    bool mouse_release_event(Platform::Application::MouseEvent &event);
    bool mouse_move_event(Platform::Application::MouseMoveEvent &event);
    bool mouse_scroll_event(Platform::Application::MouseScrollEvent &event);
    bool text_input_event(Platform::Application::TextInputEvent &event);

    void close_menu() { show_menu = false; }

private:
    void draw_add_window(const Vector2i &window_size);
    void draw_menu_window(const Vector2i &window_size, UiData &ui);
    void draw_scale_window(UiTransData &ui);
    void draw_color_window(UiData &ui);

    void draw_open_file(UiParserData &ui);
    void hover_info(const std::string &text);

    ImGuiIntegration::Context context{ NoCreate };

    ImFont *p_font;
    GL::Texture2D font_texture;

    bool show_menu;
    bool show_scale;
    bool show_color;

    ImGui::FileBrowser open_file;
};

#endif // #ifndef UI_IMGUI_H
