
#ifndef UI_MENU_H
#define UI_MENU_H

#include "ui_color.h"
#include "ui_export.h"
#include "ui_load.h"
#include "ui_scale.h"
#include "ui_train.h"

class Application;

struct uiMenu
{
    uiLoader loader;
    uiExporter exporter;
    uiScaler scaler;
    uiTrainingSettings training_set;
    uiColorSettings color_set;

    uiMenu();
    void render(Application &app);
    void close_menu() { show_menu = false; }

private:
    void draw_menu_window(const Vector2i &window_size);

    bool show_menu;
};

#include "application.h"

#endif
