
#include "application.h"
#include <exception>

uiStorer::uiStorer()
  : show_window(false)
{}

void
uiStorer::render(Application &app)
{
    if (!show_window)
        return;

    // TODO what to store?
}
