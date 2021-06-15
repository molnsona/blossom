#ifndef SIMULATION_H
#define SIMULATION_H

#include "app/state.h"

class Simulation
{
public:
    Simulation(State* p_state);
    
    void update(State* p_state);
private:
    std::vector<Vector2i> _forces;
};

#endif // #ifndef SIMULATION_H