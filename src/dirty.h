
#ifndef DIRTY_H
#define DIRTY_H

#include <cstddef>
#include <tuple>

/** \brief A piece of dirt for dirtying the caches.
 *
 * This is supposed to be a part of classes that are cached elsewhere; having
 * the `dirty` counter allows us to reliably observe if someting changed and
 * run the updates accordingly. Use with `Cleaner` to observe the changes.
 */
struct Dirt
{
    int dirt;

    Dirt()
      : dirt(0)
    {}
    /** \brief Make the cache dirty
     *
     * Call this if something changed and the caches need to be refreshed
     */
    void touch() { ++dirt; }
};

/** A piece of cache that keeps track of the dirty status.
 *
 * This is a natural counterpart of the `Dirt` class.
 */
struct Cleaner
{
    int cleaned;

    Cleaner()
      : cleaned(-1)
    {}

    /** \brief Returns true if the cache needs to be refreshed */
    bool dirty(const Dirt &d)
    {
        // handle overflows!
        return d.dirt - cleaned > 0;
    }

    /** \brief Call this when the cache is refreshed. */
    void clean(const Dirt &d)
    {
        if (cleaned < d.dirt)
            cleaned = d.dirt;
    }
};

/** \brief Multi-piece cache-dirtying object.
 *
 * This is to be inherited into data objects that are cached elsewhere, but
 * have more parts that may be transformed independently. Use with `Sweeper`.
 *
 * User structures are responsible for filling in `n` correctly.
 */
struct Dirts : public Dirt
{
    size_t n; /// Number of objects that should be cached.
    Dirts(size_t n = 0)
      : n(n)
    {}
};

/** A piece of multi-object cache.
 *
 * This is the counterpart of `Dirts`.
 */
struct Sweeper : public Cleaner
{
    size_t begin, dirts;

    Sweeper()
      : begin(0)
      , dirts(0)
    {}

    /** \brief Force-refresh the whole range */
    void refresh(const Dirts &d) { dirts = d.n; }
    void refresh(size_t n_dirts) { dirts = n_dirts; }

    /** \brief Find the range to refresh
     *
     * Return the beginning index and size of the range that needs to be
     * refreshed. Note that the range is cyclic!
     */
    std::tuple<size_t, size_t> dirty_range(const Dirts &d)
    {
        if (dirty(d)) {
            refresh(d);
            clean(d);
        }
        if (begin >= d.n)
            begin = 0;
        return { begin, dirts };
    }

    /** \brief Clean a range of the cache
     *
     * Call this to tell the Sweeper that you refresh `n` cache elements,
     * starging at the beginning index returned from dirty_range().
     * The indexing is cyclic modulo `d.n`.
     */
    void clean_range(const Dirts &d, size_t n)
    {
        if (n > dirts)
            n = dirts;
        dirts -= n;
        if (!dirts)
            begin = 0;
        else {
            begin += n;
            if (begin >= d.n)
                begin -= d.n;
        }
    }
};

#endif
