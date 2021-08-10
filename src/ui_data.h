#ifndef UI_DATA_H
#define UI_DATA_H

#include "ui_parser_data.h"
#include "ui_trans_data.h"

struct UiData
{
    UiTransData trans_data;
    UiParserData parser_data;

    bool reset{ false };

    UiData()
      : reset(false)
    {}

    void reset_data()
    {
        trans_data.reset_data();
        parser_data.reset_data();
        reset = false;
    }
};

#endif // #ifndef UI_DATA_H
