
#include "state.h"
#include "embedsom.h"

#include <iostream>

std::size_t counter = 0;

void
State::update(float time)
{
    if (ui.reset) {
        data = DataModel();
        landmarks = LandmarkModel();
        scatter = ScatterModel();
        layout_data = GraphLayoutData();
        ui.reset = false;
    }

    if (ui.parse) {
        ui.parser->parse(ui.file_path, 1000, data.data, data.d, data.n);

        landmarks.update(data);
        ui.parse = false;
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

#ifdef NO_CUDA
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
#else
    // these methods should be called only once in the initialization and then
    // only when the data/paramters change
    esom_cuda.setDim(data.d);
    esom_cuda.setK(10);
    esom_cuda.setPoints(data.n, data.data.data());
    esom_cuda.setLandmarks(landmarks.lodim_vertices.size(),
                           landmarks.hidim_vertices.data(),
                           landmarks.lodim_vertices[0].data());

    // this is the actual method that is called in every update
    esom_cuda.embedsom(
      2.0, 0.2, scatter.points[0].data()); // boost and adjust parameters are
                                           // now passed in every call, but we
                                           // might want to cache them iside?

    if (counter == 9) {
        std::cout << esom_cuda.getAvgPointsUploadTime() << "ms \t"
                  << esom_cuda.getAvgLandmarksUploadTime() << "ms \t"
                  << esom_cuda.getAvgProcessingTime() << "ms" << std::endl;
    }
    counter = (counter + 1) % 10;
#endif
}
