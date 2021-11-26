
#include "pnorm.h"

#include <cfloat>
#include <cmath>

/* pnorm() is taken and adapted from R-4.0.2, originally licensed under GPLv2,
 * compatible with BlosSOM. We whole-heartedly thank the R project for
 * maintaining a quality library of mathematical function implementations.
 * Original licence and copyright follows:
 *
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 1998	    Ross Ihaka
 *  Copyright (C) 2000-2013 The R Core Team
 *  Copyright (C) 2003	    The R Foundation
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/
 */

#define SIXTEN 16
#define M_1_SQRT_2PI 0.398942280401432677939946059934
#define M_SQRT_32 5.656854249492380195206754896838

static void
pnorm_both(double x, double &cum, double &ccum)
{
    const static double a[5] = { 2.2352520354606839287,
                                 161.02823106855587881,
                                 1067.6894854603709582,
                                 18154.981253343561249,
                                 0.065682337918207449113 };
    const static double b[4] = { 47.20258190468824187,
                                 976.09855173777669322,
                                 10260.932208618978205,
                                 45507.789335026729956 };
    const static double c[9] = { 0.39894151208813466764,  8.8831497943883759412,
                                 93.506656132177855979,   597.27027639480026226,
                                 2494.5375852903726711,   6848.1904505362823326,
                                 11602.651437647350124,   9842.7148383839780218,
                                 1.0765576773720192317e-8 };
    const static double d[8] = { 22.266688044328115691, 235.38790178262499861,
                                 1519.377599407554805,  6485.558298266760755,
                                 18615.571640885098091, 34900.952721145977266,
                                 38912.003286093271411, 19685.429676859990727 };
    const static double p[6] = {
        0.21589853405795699,     0.1274011611602473639, 0.022235277870649807,
        0.001421619193227893466, 2.9112874951168792e-5, 0.02307344176494017303
    };
    const static double q[5] = { 1.28426009614491121,
                                 0.468238212480865118,
                                 0.0659881378689285515,
                                 0.00378239633202758244,
                                 7.29751555083966205e-5 };

    double xden, xnum, temp, del, eps, xsq, y;
    int i;

    eps = DBL_EPSILON * 0.5;

    constexpr int lower = 1;
    constexpr int upper = 0;

    y = fabs(x);
    if (y <= 0.67448975) {
        if (y > eps) {
            xsq = x * x;
            xnum = a[4] * xsq;
            xden = xsq;
            for (i = 0; i < 3; ++i) {
                xnum = (xnum + a[i]) * xsq;
                xden = (xden + b[i]) * xsq;
            }
        } else
            xnum = xden = 0.0;

        temp = x * (xnum + a[3]) / (xden + b[3]);
        if (lower)
            cum = 0.5 + temp;
        if (upper)
            ccum = 0.5 - temp;
    } else if (y <= M_SQRT_32) {
        xnum = c[8] * y;
        xden = y;
        for (i = 0; i < 7; ++i) {
            xnum = (xnum + c[i]) * y;
            xden = (xden + d[i]) * y;
        }
        temp = (xnum + c[7]) / (xden + d[7]);

#define do_del(X)                                                              \
    xsq = trunc(X * SIXTEN) / SIXTEN;                                          \
    del = (X - xsq) * (X + xsq);                                               \
    cum = exp(-xsq * xsq * 0.5) * exp(-del * 0.5) * temp;                      \
    ccum = 1.0 - cum;

#define swap_tail                                                              \
    if (x > 0.) {                                                              \
        temp = cum;                                                            \
        if (lower)                                                             \
            cum = ccum;                                                        \
        ccum = temp;                                                           \
    }

        do_del(y);
        swap_tail;
    }

    else if ((lower && -37.5193 < x && x < 8.2924) ||
             (upper && -8.2924 < x && x < 37.5193)) {

        xsq = 1.0 / (x * x);
        xnum = p[5] * xsq;
        xden = xsq;
        for (i = 0; i < 4; ++i) {
            xnum = (xnum + p[i]) * xsq;
            xden = (xden + q[i]) * xsq;
        }
        temp = xsq * (xnum + p[4]) / (xden + q[4]);
        temp = (M_1_SQRT_2PI - temp) / y;

        do_del(x);
        swap_tail;
    } else {
        if (x > 0) {
            cum = 1;
            ccum = 0;
        } else {
            cum = 0;
            ccum = 1;
        }
    }

    if (cum < DBL_MIN)
        cum = 0.;
    if (ccum < DBL_MIN)
        ccum = 0.;
    return;
}

float
pnormf(float x, float mean, float sd)
{
    double p, cp;

    if (!std::isfinite(x) && mean == x)
        return NAN;
    if (sd <= 0) {
        if (sd < 0)
            return 0.5;
        return (x < mean) ? 0 : 1;
    }
    p = (x - mean) / sd;
    if (!std::isfinite(p))
        return (x < mean) ? 0 : 1;
    x = p;

    pnorm_both(x, p, cp);

    return p;
}
