/*
The MIT License

Copyright (c) 2021 Adam Smelko
                   Mirek Kratochvil

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and
associated documentation files (the "Software"), to
deal in the Software without restriction, including
without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom
the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef CUDA_STRUCTS_H
#define CUDA_STRUCTS_H

#include "cuda_runtime.h"

#include <cfloat>
#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

/** A structure for packing neighbor index and distance for kNN search. */
template<typename F>
struct knn_entry
{
    F distance;
    uint32_t index;

    CUDA_CALLABLE_MEMBER bool operator<(const knn_entry &rhs) const
    {
        return distance < rhs.distance ||
               (distance == rhs.distance && index < rhs.index);
    }

    CUDA_CALLABLE_MEMBER bool operator>(const knn_entry &rhs) const
    {
        return rhs < *this;
    }

    CUDA_CALLABLE_MEMBER bool operator<=(const knn_entry &rhs) const
    {
        return !(rhs < *this);
    }

    CUDA_CALLABLE_MEMBER bool operator>=(const knn_entry &rhs) const
    {
        return !(*this < rhs);
    }
};

template<unsigned N, typename T>
struct Vec
{};
template<>
struct Vec<2, float>
{
    using Type = float2;
};
template<>
struct Vec<4, float>
{
    using Type = float4;
};
template<>
struct Vec<2, double>
{
    using Type = double2;
};
template<>
struct Vec<4, double>
{
    using Type = double4;
};

template<typename F>
constexpr F valueMax;

template<>
constexpr float valueMax<float> = FLT_MAX;
template<>
constexpr double valueMax<double> = DBL_MAX;

#endif
