
#include "state.h"

void
State::update(float time)
{
    graph_layout_step(
      layout_data, model.vertices, model.edges, model.edge_lengths, time);
}
