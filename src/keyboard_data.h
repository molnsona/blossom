#ifndef KEYBOARD_DATA_H
#define KEYBOARD_DATA_H

#include <Magnum/Magnum.h>
//#include <Magnum/Math/Vector2i.h>
#include <Magnum/Math/Vector2.h>

using namespace Magnum;

struct KeyboardData
{
    bool ctrl_pressed;

    KeyboardData()
      : ctrl_pressed(false)
    {}
};

#endif // #ifndef KEYBOARD_DATA_H
