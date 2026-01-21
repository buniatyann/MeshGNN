# MeshGNN Documentation

Welcome to the MeshGNN documentation. This guide provides comprehensive information about all modules, their implementations, and how they work together.

## Overview

MeshGNN is a C++17 library implementing Graph Neural Networks for 3D mesh simplification. It processes triangular meshes, extracts geometric features, and applies GNN-based edge collapse algorithms for mesh decimation.

### Key Features

- **No External Dependencies**: Pure C++17 standard library implementation
- **Complete Pipeline**: From OBJ loading to learned mesh simplification
- **Trainable GNN**: SGD and Adam optimizers with MSE/cross-entropy loss
- **Efficient Sparse Operations**: CSR format for memory-efficient graph computations
- **Parallel Execution**: Automatic vectorization using `std::execution::par_unseq`

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              MeshGNN Library                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │    Core     │    │    Math     │    │  Geometry   │    │   GNN    │ │
│  │             │    │             │    │             │    │          │ │
│  │ • types     │    │ • vector    │    │ • mesh      │    │ • layers │ │
│  │ • config    │───▶│ • dense_mat │───▶│ • obj_load  │───▶│ • pipe   │ │
│  │ • random    │    │ • sparse_mat│    │ • features  │    │ • train  │ │
│  │             │    │             │    │ • processor │    │ • optim  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
│                                                                         │
│                            ┌─────────────┐                              │
│                            │    Graph    │                              │
│                            │             │                              │
│                            │ • structure │                              │
│                            │ • features  │                              │
│                            │ • adjacency │                              │
│                            └─────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

```
OBJ File
    │
    ▼
┌─────────────────────────────────────┐
│  OBJ Loader                         │
│  • Parse vertices, faces, normals   │
│  • Triangulate polygons             │
│  • Generate missing normals         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Mesh                               │
│  • Store topology (V, E, F)         │
│  • Build adjacency structures       │
│  • Edge index mapping               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Feature Extraction                 │
│  • Node: coords + normals + curv    │
│  • Edge: length + normal angle      │
│  • Gaussian curvature computation   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Graph                              │
│  • Structured node/edge features    │
│  • Sparse CSR adjacency matrix      │
│  • Message passing support          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  GNN Pipeline                       │
│  • GCN / EdgeConv layers            │
│  • Forward propagation              │
│  • Learn edge collapse priorities   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Mesh Processor                     │
│  • Priority queue simplification    │
│  • GNN-driven edge selection        │
│  • Topology-preserving collapse     │
└─────────────────────────────────────┘
    │
    ▼
Simplified Mesh
```

## Module Documentation

### Core Infrastructure
- [Core Module](modules/core.md) - Types, configuration, and random number generation

### Mathematical Operations
- [Math Module](modules/math.md) - Vector operations, dense and sparse matrices

### Geometry Processing
- [Geometry Module](modules/geometry.md) - Mesh representation, OBJ loading, feature extraction, mesh simplification

### Graph Representation
- [Graph Module](modules/graph.md) - Graph data structure for GNN input

### Neural Networks
- [GNN Module](modules/gnn.md) - Layers, pipeline, training, optimizers, and loss functions

### Complete Reference
- [API Reference](modules/api_reference.md) - Full API documentation with all functions and classes

## Namespace Structure

All code lives under the `gnnmath::` namespace with sub-namespaces:

```cpp
gnnmath::
├── core/
│   ├── types.hpp      // scalar_t, index_t, feature_t
│   ├── config.hpp     // Constants and version info
│   └── random.hpp     // Thread-safe RNG
│
├── math/
│   ├── vector.hpp     // Vector operations and activations
│   ├── dense_matrix.hpp   // Dense matrix class
│   └── sparse_matrix.hpp  // CSR sparse matrix
│
├── geometry/
│   ├── mesh.hpp           // Mesh representation
│   ├── obj_loader.hpp     // OBJ file parsing
│   ├── feature_extraction.hpp  // Geometric features
│   └── mesh_processor.hpp      // Simplification
│
├── graph.hpp              // Graph structure
│
└── gnn/
    ├── layers/
    │   ├── layer.hpp      // Abstract base
    │   ├── gcn_layer.hpp  // Graph convolution
    │   └── edge_conv_layer.hpp  // Edge convolution
    │
    ├── optimizers/
    │   ├── optimizer.hpp  // Abstract base
    │   ├── sgd.hpp        // SGD optimizer
    │   └── adam.hpp       // Adam optimizer
    │
    ├── losses/
    │   ├── loss.hpp       // Abstract base
    │   ├── mse.hpp        // Mean squared error
    │   └── cross_entropy.hpp  // Cross-entropy
    │
    ├── pipeline.hpp       // Layer stacking
    └── training.hpp       // Trainer class
```

## Quick Start

### Building

```bash
# CMake build (recommended)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# Run the demo
./meshgnn_demo ../data/Porsche_911_GT2.obj
```

### Basic Usage

```cpp
#include <gnnmath/gnnmath.hpp>

int main() {
    // Load mesh
    gnnmath::mesh m;
    m.load_obj("model.obj");

    // Extract features
    auto node_features = gnnmath::compute_node_features(m);
    auto edge_features = gnnmath::compute_edge_features(m);

    // Create graph
    auto g = gnnmath::graph::from_mesh(m);
    auto adj = g.to_adjacency_matrix();

    // Build GNN pipeline
    auto pipeline = std::make_shared<gnnmath::gnn::pipeline>();
    pipeline->add_layer(std::make_unique<gnnmath::gnn::gcn_layer>(7, 32));
    pipeline->add_layer(std::make_unique<gnnmath::gnn::gcn_layer>(32, 16));
    pipeline->add_layer(std::make_unique<gnnmath::gnn::gcn_layer>(16, 1));

    // Process through pipeline
    auto scores = pipeline->process(node_features, adj);

    // Simplify mesh using learned scores
    gnnmath::simplify_with_gnn_scores(m, target_vertices, scores);

    return 0;
}
```

## Design Principles

1. **Header-Only Validation**: Dimension checks and error handling throughout
2. **Parallel Execution**: Uses `std::execution::par_unseq` for automatic vectorization
3. **Overflow Prevention**: Careful clamping in exponential operations
4. **CSR Efficiency**: Sparse matrices for O(nnz) operations
5. **Lazy Deletion**: Priority queue with versioning for O(E log E) simplification
6. **Polymorphism**: Abstract base classes for layers, optimizers, and losses
7. **Move Semantics**: Efficient data transfer without copying
8. **Thread Safety**: Thread-local random number generator

## Requirements

- **Compiler**: C++17 (GCC 7+, Clang 5+)
- **Optional**: Intel TBB for parallel execution policies
