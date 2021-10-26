#ifndef UI_DATA_H
#define UI_DATA_H

#include "ui_parser_data.h"
#include "ui_sliders_data.h"
#include "ui_trans_data.h"

struct UiData
{
    UiTransData trans_data;
    UiParserData parser_data;
    UiSlidersData sliders_data;

    int color_ind;

    bool reset{ false };

    UiData()
      : color_ind(0)
      , reset(false)
    {}

    void reset_data()
    {
        trans_data.reset_data();
        parser_data.reset_data();
        sliders_data.reset_data();
        color_ind = 0;
        reset = false;
    }
};

#endif // #ifndef UI_DATA_H
