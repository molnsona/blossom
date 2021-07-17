#include <Magnum/Math/Functions.h>

#include <iostream>

#include "simulation.h"

using namespace Magnum;
using namespace Math::Literals;

Simulation::Simulation(State *p_state)
{
    _forces.resize(p_state->vtx_pos.size());
}

void
Simulation::update(State *p_state)
{
    // float slowdown = 0.995f;

    // // if(p_state->time < p_state->timeout)
    // // {
    //     for(auto&& edge: p_state->edges)
    //     {
    //         Vector2 v1_pos, v2_pos;
    //         int e_len, error;
    //         float resistance = 0.75f;

    //         v1_pos = p_state->vtx_pos[edge.x()];
    //         v2_pos = p_state->vtx_pos[edge.y()];

    //         e_len = abs(v1_pos - v2_pos).length();
    //         //if(e_len == 100) std::cout << e_len;
    //         error = e_len - p_state->expected_len;
    //         _forces[edge.x()] += error * (v2_pos - v1_pos) * resistance;
    //         _forces[edge.y()] += error * (v1_pos - v2_pos) * resistance;
    //     }

    //     for (size_t i = 0; i < p_state->vtx_pos.size(); ++i)
    //     {
    //         p_state->vtx_pos[i].x() = (p_state->vtx_pos[i].x() + 0.0001 *
    //         _forces[i].x()) * slowdown; p_state->vtx_pos[i].y() =
    //         (p_state->vtx_pos[i].y() + 0.0001 * _forces[i].y()) * slowdown;

    //     }

    //     ++p_state->time;
    // // }

    if (p_state->time < p_state->timeout) {
        for (auto &&edge : p_state->edges) {
            Vector2 v1_pos, v2_pos;
            int e_len, error;

            v1_pos = p_state->vtx_pos[edge.x()];
            v2_pos = p_state->vtx_pos[edge.y()];

            e_len = abs(v1_pos - v2_pos).length();
            error = atan(e_len - p_state->expected_len);
            _forces[edge.x()] += error * (v2_pos - v1_pos);
            _forces[edge.y()] += error * (v1_pos - v2_pos);
        }

        for (size_t i = 0; i < p_state->vtx_pos.size(); ++i)
            for (size_t j = i + 1; j < p_state->vtx_pos.size(); ++j) {

                Vector2 d = p_state->vtx_pos[i] - p_state->vtx_pos[j];

                auto f = exp(-d.length() / 50);
                _forces[i] += d * f;
                _forces[j] += d * -f;
            }

        for (size_t i = 0; i < p_state->vtx_pos.size(); ++i) {
            p_state->vtx_pos[i] += 0.0001 * _forces[i];
        }
        ++p_state->time;
    }
}