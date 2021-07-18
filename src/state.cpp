
#include "state.h"

void
State::update(float time)
{
    graph_layout_step(layout_data,
                      landmarks.vertices,
                      landmarks.edges,
                      landmarks.edge_lengths,
                      time);
}
