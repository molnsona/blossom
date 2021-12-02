#ifndef CUDA_STRUCTS_H
#define CUDA_STRUCTS_H

#include "cuda_runtime.h"

#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

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
struct V
{};
template<>
struct V<2, float>
{
    using Type = float2;
};
template<>
struct V<4, float>
{
    using Type = float4;
};
template<>
struct V<2, double>
{
    using Type = double2;
};
template<>
struct V<4, double>
{
    using Type = double4;
};

#endif
