# BlosSOM architecture

## Technologies
BlosSOM is written in C++ using [GLFW](https://www.glfw.org/) for rendering and [Dear ImGUI](https://github.com/ocornut/imgui) for graphical user interface. The [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone) platform is used by EmbedSOM algorithm for GPU computations.

## Render cycle
This application has graphical output that is rendered in cycles. One cycle looks as follows (`main()` method in the `main.cpp` file): 
```cpp
timer.tick();
view.update();
state.update();
renderer.update();
imgui.render();
```

### State update
The main algorithm updates happen in the State update cycle that looks as follows (`State::update()` method in the `state.h` file):
```cpp
stats.update();
trans.update();
scaled.update();

if(kmeans) kmeans_landmark_step();
if(knn_edges) make_knn_edges();
if(graph_layout) graph_layout_step();
if(tsne) tsne_layout_step();
if(som) som_landmark_step();

colors.update();
scatter.update();
```  

For further information about classes and files see [Doxygen documentation](https://molnsona.github.io/blossom/) of the public attributes of the Application and State classes.
