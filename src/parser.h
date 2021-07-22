#ifndef PARSER_H
#define PARSER_H

class Parser
{
public:
    virtual void parse(const std::string &file_path,
                       size_t points_count,
                       std::vector<float> &out_data,
                       size_t &dim,
                       size_t &n) = 0;
    virtual ~Parser() {}
};

#endif // #ifndef PARSER_H
