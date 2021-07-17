
#include "state.h"

void
State::update(float time)
{
    graph_layout_step(layout_data, vtx_pos, edges, lengths, time);
}
