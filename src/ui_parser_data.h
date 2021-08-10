#ifndef UI_PARSER_DATA_H
#define UI_PARSER_DATA_H

#include <memory>

#include "parser.h"

struct UiParserData
{
    bool parse{ false };
    std::string file_path;
    std::unique_ptr<Parser> parser;

    bool is_tsv = false; // TODO: Remove when landmarks are dynamically computed

    void reset_data()
    {
        parse = false;
        parser = nullptr;
        is_tsv = false; // TODO: Remove when landmarks are dynamically computed
    }
};

#endif // #ifndef UI_PARSER_DATA_H
