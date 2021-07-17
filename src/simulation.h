#ifndef SIMULATION_H
#define SIMULATION_H

#include "state.h"

class Simulation
{
public:
    Simulation() = delete;
    Simulation(State *p_state);

    void update(State *p_state);

private:
    std::vector<Vector2> _forces;
};

/**
 * \brief Serial implementation of the physical model.
 * \tparam F Floating point num type used for point coordinates (float or
 * double). \tparam IDX_T Type used for various indices (e.g., referencing
 * vertices in graph). \tparam LEN_T Type in which lengths of edges is
 * represented.
 */
class SerialSimulator
{
public:
    typedef float coord_t;  // Type of point coordinates.
    typedef coord_t real_t; // Type of additional float parameters.
    typedef std::uint32_t index_t;
    typedef int length_t;
    typedef Vector2 point_t;
    typedef Vector2i edge_t;

private:
    const std::vector<edge_t> &mEdges; ///< Reference to the graph edges.
    const std::vector<length_t>
      &mLengths; ///< Reference to the graph lengths of the edges.
    std::vector<point_t> mVelocities; ///< Point velocity vectors.
    std::vector<point_t> mForces; ///< Preallocated buffer for force vectors.

    float vertexRepulsion{ 0.1f };
    float edgeCompulsion{ 20.0f };
    float timeQuantum{ 0.001f };
    float vertexMass{ 1.0f };
    float slowdown{ 0.95 };

    /**
     * \brief Add repulsive force that affects selected points.
     *		This function updates internal array mForces.
     * \param points Current point coordinates.
     * \param p1 One of the points for which the repulsive force is computed.
     * \param p2 One of the points for which the repulsive force is computed.
     * \param forces Vector where forces affecting points are being accumulated.
     */
    void addRepulsiveForce(const std::vector<point_t> &points,
                           index_t p1,
                           index_t p2,
                           std::vector<point_t> &forces)
    {
        auto d = points[p2] - points[p1];
        auto q = exp(-d.length() / 50) * 1000;
        forces[p1] += -q * d;
        forces[p2] += q * d;
#if martin
        real_t dx = (real_t)points[p1].x() - (real_t)points[p2].x();
        real_t dy = (real_t)points[p1].y() - (real_t)points[p2].y();
        real_t sqLen = std::max<real_t>(dx * dx + dy * dy, (real_t)0.0001);
        real_t fact =
          vertexRepulsion / (sqLen * (real_t)std::sqrt(sqLen)); // mul factor
        dx *= fact;
        dy *= fact;
        forces[p1].x() += dx;
        forces[p1].y() += dy;
        forces[p2].x() -= dx;
        forces[p2].y() -= dy;
#endif
    }

    /**
     * \brief Add compulsive force that affects selected points connected with
     *an edge. This function updates internal array mForces. \param points
     *Current point coordinates. \param p1 One of the points adjacent to the
     *edge. \param p2 One of the points adjacent to the edge. \param length
     *Length of the edge. \param forces Vector where forces affecting points are
     *being accumulated.
     */
    void addCompulsiveForce(const std::vector<point_t> &points,
                            index_t p1,
                            index_t p2,
                            length_t length,
                            std::vector<point_t> &forces)
    {
        auto d = points[p2] - points[p1];
        auto q = (length - d.length()) * 5;
        forces[p1] += -q * d;
        forces[p2] += q * d;
#if martin
        real_t dx = (real_t)points[p2].x() - (real_t)points[p1].x();
        real_t dy = (real_t)points[p2].y() - (real_t)points[p1].y();
        real_t sqLen = dx * dx + dy * dy;
        real_t fact =
          (real_t)std::sqrt(sqLen) * edgeCompulsion / (real_t)(length);
        dx *= fact;
        dy *= fact;
        forces[p1].x() += dx;
        forces[p1].y() += dy;
        forces[p2].x() -= dx;
        forces[p2].y() -= dy;
#endif
    }

    /**
     * \brief Update velocities based on current forces affecting the points.
     */
    void updateVelocities(State *p_state, const std::vector<point_t> &forces)
    {
        real_t fact =
          timeQuantum / vertexMass; // v = Ft/m  => t/m is mul factor for F.
        for (std::size_t i = 0; i < mVelocities.size(); ++i) {
            // if(p_state->vtx_selected && (i == p_state->vtx_ind)) continue;
            mVelocities[i].x() =
              (mVelocities[i].x() + (real_t)forces[i].x() * fact) * slowdown;
            mVelocities[i].y() =
              (mVelocities[i].y() + (real_t)forces[i].y() * fact) * slowdown;
        }
    }

public:
    SerialSimulator(index_t pointCount,
                    const std::vector<edge_t> &edges,
                    const std::vector<length_t> &lengths)
      : mEdges(edges)
      , mLengths(lengths)
    {
        mVelocities.resize(pointCount);
        mForces.resize(pointCount);
    }

    /**
     * \brief Reset velocities to zero.
     */
    void resetVelocities()
    {
        for (std::size_t i = 0; i < mVelocities.size(); ++i)
            mVelocities[i].x() = mVelocities[i].y() = (real_t)0.0;
    }

    /**
     * \brief Return current velocities.
     */
    const std::vector<point_t> &getVelocities() const { return mVelocities; }

    /**
     * \brief Override internal velocities with velocities in given buffer.
     * \param velocities The buffer with new velocities.
     *		The buffer must have the same size as the internal buffer.
     */
    void setVelocities(const std::vector<point_t> &velocities)
    {
        mVelocities = velocities;
    }

    /**
     * \brief Swap internal velocities buffer with given buffer.
     * \param velocities The buffer with new velocities.
     *		The buffer must have the same size as the internal buffer.
     */
    void swapVelocities(std::vector<point_t> &velocities)
    {
        // if (mVelocities.size() != velocities.size())
        // 	throw (bpp::RuntimeError() << "Cannot swap internal velocity buffer
        // with a buffer of a different size."
        // 	<< "Current model uses " << mVelocities.size() << " points, but the
        // buffer has " << velocities.size() << " points.");
        mVelocities.swap(velocities);
    }

    void computeForces(State *p_state,
                       std::vector<point_t> &points,
                       std::vector<point_t> &forces)
    {
        forces.resize(points.size());

        // Clear forces array for another run.
        for (std::size_t i = 0; i < forces.size(); ++i) {
            forces[i].x() = forces[i].y() = (real_t)0.0;
        }

        // Compute repulsive forces between all vertices.
        for (index_t i = 1; i < forces.size(); ++i) {
            for (index_t j = 0; j < i; ++j)
                addRepulsiveForce(points, i, j, forces);
        }

        // Compute compulsive forces of the edges.
        for (std::size_t i = 0; i < mEdges.size(); ++i)
            addCompulsiveForce(
              points, mEdges[i].x(), mEdges[i].y(), mLengths[i], forces);
    }

    /**
     * \brief The main method that computes another version of point positions.
     * \param points Point positions that are updated by the function.
     * \note The function updates internal array with velocities.
     */
    void updatePoints(State *p_state)
    {
        // if (points.size() != mVelocities.size())
        // 	throw (bpp::RuntimeError() << "Cannot compute next version of point
        // positions."
        // 		<< "Current model uses " << mVelocities.size() << " points, but
        // the given buffer has " << points.size() << " points.");

        computeForces(p_state, p_state->vtx_pos, mForces);
        updateVelocities(p_state, mForces);

        // Update point positions.
        for (std::size_t i = 0; i < mVelocities.size(); ++i) {
            // if(p_state->vtx_selected && (i == p_state->vtx_ind)) continue;
            p_state->vtx_pos[i].x() += mVelocities[i].x() * timeQuantum;
            p_state->vtx_pos[i].y() += mVelocities[i].y() * timeQuantum;
        }
    }
};

#endif // #ifndef SIMULATION_H
