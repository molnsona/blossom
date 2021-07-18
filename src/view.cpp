
#include "view.h"

void
View::set_fb_size(Vector2i s)
{
    // TODO move the other stuff (esp. zooms) to avoid weird artifacts
    fb_size = s;
}
