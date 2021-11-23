#ifndef UI_PARSER_DATA_H
#define UI_PARSER_DATA_H

#include <memory>

struct UiParserData
{
    bool parse;
    std::string file_path;

    UiParserData() { init(); }

    void init() { reset_data(); }

    void reset_data()
    {
        parse = false;
    }
};

#endif // #ifndef UI_PARSER_DATA_H
