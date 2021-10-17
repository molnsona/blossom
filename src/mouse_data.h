#ifndef MOUSE_DATA_H
#define MOUSE_DATA_H

#include <Magnum/Magnum.h>
//#include <Magnum/Math/Vector2i.h>
#include <Magnum/Math/Vector2.h>

using namespace Magnum;

struct MouseData
{
    // Raw coordinates on the screen (upper left [0,0])
    Vector2i mouse_pos;

    bool mouse_pressed;
    bool vert_pressed;
    std::size_t vert_ind;

    MouseData()
      : mouse_pressed(false)
      , vert_pressed(false)
      , vert_ind(0)
    {}
};

#endif // #ifndef MOUSE_DATA_H