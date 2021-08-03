#ifndef UI_DATA_H
#define UI_DATA_H

struct UiData
{
    int cell_cnt{ 10000 };
    int mean{ 0 };
    int std_dev{ 300 };

    bool parse{ false };
    std::string file_path;
    std::unique_ptr<Parser> parser;
    bool reset{ false };
};

#endif // #ifndef UI_DATA_H
