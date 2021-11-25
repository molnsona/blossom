#include "trans_data.h"

void
TransData::update(const DataModel &dm)
{
    if (config.size() != dm.d)
        reset(dm);
}

void
TransData::reset(const DataModel &dm)
{
    config.clear();
    data.clear();
    n = dm.n;
    d = dm.d;
    data = dm.data;
    config.resize(d);
}

void
TransData::disable_col(size_t c)
{
    // TODO update config, remove the column from output if needed, reduce `d`,
    // ...
}

void
TransData::enable_col(size_t c)
{
    // TODO reverse of disable_col
}
