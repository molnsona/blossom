#ifndef UI_PARSER_DATA_H
#define UI_PARSER_DATA_H

#include <memory>

struct UiParserData
{
    bool parse;
    std::string file_path;

    bool is_tsv; // TODO: Remove when landmarks are dynamically computed

    UiParserData() { init(); }

    void init() { reset_data(); }

    void reset_data()
    {
        parse = false;
        is_tsv = false; // TODO: Remove when landmarks are dynamically computed
    }
};

#endif // #ifndef UI_PARSER_DATA_H
