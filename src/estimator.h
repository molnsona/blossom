/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Sona Molnarova
 *
 * BlosSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * BlosSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * BlosSOM. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <array>
#include <cstddef>
#include <tuple>

typedef std::array<float, 4> mat2x2;
typedef std::array<float, 2> mat2x1;

class Estimator
{
public:
    Estimator();
    void process_measurement(size_t n, float t);
    std::tuple<float, float> get_estimate();
    float get_z(float x, float y);
    float get_var();
    mat2x2 get_cov_matrix();
    void reset();

private:
    float a;
    float b;
    float c;
    float d;
    float e;
    float f;
    float alpha;
    float coalpha;
};

#endif // #ifndef ESTIMATOR_H
