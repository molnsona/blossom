
#include "state.h"
#include "embedsom.h"

void
State::update(float time)
{
    if (parse) {
        fcs_parser.parse(file_path, data.data, data.d, data.n);

        landmarks.update(data);
        parse = false;
    }

    graph_layout_step(layout_data,
                      mouse,
                      landmarks.lodim_vertices,
                      landmarks.edges,
                      landmarks.edge_lengths,
                      time);

    if (scatter.points.size() != data.n) {
        scatter.points.clear();
        scatter.points.resize(data.n);
    }

    // TODO check that data dimension matches landmark dimension and that
    // model sizes are matching (this is going to change dynamically)
    embedsom(data.n,
             landmarks.lodim_vertices.size(),
             data.d, // should be the same as landmarks.d
             2.0,
             10,
             0.2,
             data.data.data() /* <3 */,
             landmarks.hidim_vertices.data(),
             landmarks.lodim_vertices[0].data(),
             scatter.points[0].data());
}
