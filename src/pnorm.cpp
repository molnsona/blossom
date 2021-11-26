
#include "pnorm.h"

float
pnormf(float x, float mean, float sd)
{
    // TODO this is really poor.
    x -= mean;
    x /= 2 * sd;
    x += 0.5;
    return x > 1 ? 1 : x < 0 ? 0 : x;
}
