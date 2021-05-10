#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <cmath>
#include <limits>

#include "ui/imgui_config.h"

/** Shifts from [a,b] to [c, d]*/
float shift_interval(float value, float a, float b, float c, float d) { return c + ((d - c) / (b - a)) * (value - a); }

std::vector<std::string> split(const std::string& str, char delim) {
	std::vector<std::string> result;
	std::stringstream ss(str);
	std::string item;

	while (getline(ss, item, delim)) {
		result.emplace_back(item);
	}

	return result;
}

struct Point {
    Point(int _x, int _y) : x(_x), y(_y) {}
    int x = 0;
    int y = 0;

    int get_max(int value) {
        return std::max(value, std::max(x, y));
    }
    int get_min(int value) {
        return std::min(value, std::min(x, y));
    }
};

// Normal distribution
std::pair<int, int> fill_vector(std::vector<Point>& result) {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<> d{0,200};

    int min = std::numeric_limits<int>::max();
    int max = std::numeric_limits<int>::min();

    result.reserve(10000);
    for (size_t i = 0; i < 10000; i++)
    {
        Point tmp_point = Point{std::round(d(gen)), std::round(d(gen))};
        min = tmp_point.get_min(min);
        max = tmp_point.get_max(max);
        result.emplace_back(Point{std::round(d(gen)), std::round(d(gen))});
    } 

    return {min, max};
}


void fill_pixels(std::vector<unsigned char>& pixels) {
    std::vector<Point> points;
    auto [min, max] = fill_vector(points);

    for(auto&& point: points) {
        if(point.x > max) point.x = max;
        if(point.x < min) point.x = min;
        if(point.y > max) point.y = max;
        if(point.y < min) point.y = min;
        std::size_t x_plot = (std::size_t)shift_interval(
            point.x, 
            min,
            max,
            0,
            PLOT_WIDTH
            );
        std::size_t y_plot = (std::size_t)shift_interval(
            point.y, 
            min,
            max,
            0,
            PLOT_HEIGHT
            );

        size_t index = BYTES_PER_PIXEL * (x_plot + y_plot * PLOT_WIDTH);
        //size_t index = 39999;
        pixels[index] = 0;
        pixels[index + 1] = 0;
        pixels[index + 2] = 0;
    }
}


#endif // #ifndef UTILS_HPP