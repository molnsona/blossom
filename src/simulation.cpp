#include <Magnum/Math/Functions.h>

#include "simulation.h"

using namespace Magnum;
using namespace Math::Literals;

Simulation::Simulation(State* p_state)
{
    _forces.resize(p_state->vtx_pos.size());
}

void Simulation::update(State* p_state)
{    
    if(p_state->time < p_state->timeout)
    {
        for(auto&& edge: p_state->edges)
        {
            Vector2i v1_pos, v2_pos;
            int e_len, error;

            v1_pos = p_state->vtx_pos[edge.x()];
            v2_pos = p_state->vtx_pos[edge.y()];

            e_len = abs(v1_pos - v2_pos).length();
            error = e_len - p_state->expected_len;
            _forces[edge.x()] += error * (v2_pos - v1_pos);
            _forces[edge.y()] += error * (v1_pos - v2_pos);
        }

        for (size_t i = 0; i < p_state->vtx_pos.size(); ++i)
        {
            p_state->vtx_pos[i] += 0.0001 * _forces[i];
        }
        ++p_state->time;
    }
}