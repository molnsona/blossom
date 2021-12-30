/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
 *                    Sona Molnarova
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

#ifndef GRAPH_MODEL_H
#define GRAPH_MODEL_H

#include "view.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>
#include <vector>

#include "dirty.h"

/**
 * @brief Model of the high- and low-dimensional landmarks.
 *
 */
struct LandmarkModel : public Dirt
{
    /** Dimension size. */
    size_t d;
    /** One-dimensional array storing d-dimensional landmark coordinates in
     * row-major order. */
    std::vector<float> hidim_vertices;
    /** Array storing two-dimensional landmark coordinates. */
    std::vector<Magnum::Vector2> lodim_vertices;

    /** Lengths of all edges.
     *
     * The ID of the edge is the index of the array and corresponds to @ref
     * edges indices. */
    std::vector<float> edge_lengths;
    /** Array of vertex ID pairs.
     *
     * The ID of the edge is the index of the array and corresponds to @ref
     * edge_lengths indices.
     *
     * \warning constraint: first vertex ID < second vertex ID
     */
    std::vector<std::pair<size_t, size_t>> edges; // constraint: first<second

    /**
     * @brief Creates empty landmarks with dimension 0.
     *
     */
    LandmarkModel();
    /**
     * @brief Updates current dimension and calls @ref init_grid().
     *
     * @param dim
     */
    void update_dim(size_t dim);
    /**
     * @brief Creates squared landmarks layout, without edges.
     *
     * It will create \p side * \p side two-dimensional landmarks and
     * \p side * \p side * @ref LandmarkModel::d high-dimensional landmarks.
     *
     * @param side Side of the square.
     */
    void init_grid(size_t side);

    /**
     * @brief Sets two-dimensional position of the pressed landmark to mouse
     * position.
     *
     * @param ind Index of the pressed landmark.
     * @param mouse_pos Mouse screen position.
     */
    void move(size_t ind, const Magnum::Vector2 &mouse_pos);
    /**
     * @brief Creates new landmark with the same two- and high-dimensional
     * coordinates as the given landmark.
     *
     * @param ind Index of the landmark which will be duplicated.
     */
    void duplicate(size_t ind);
    /**
     * @brief Creates new landmark with the two- and high-dimensional
     * coordinates as the closeset landmark.
     *
     * @param mouse_pos Mouse screen position.
     */
    void add(const Magnum::Vector2 &mouse_pos);
    /**
     * @brief Removes landmark and corresponding edges.
     *
     * @param ind Index of the removed landmark.
     */
    void remove(size_t ind);

    /**
     * @brief Counts closest landmark to the given position.
     *
     * @param mouse_pos Mouse screen position.
     * @return size_t Index of the closest landmark.
     */
    size_t closest_landmark(const Magnum::Vector2 &mouse_pos) const;

    /**
     * @brief Reurns number of the 2D landmarks.
     *
     * @return size_t Number of the 2D landmarks.
     */
    size_t n_landmarks() const { return lodim_vertices.size(); }
};

#endif
