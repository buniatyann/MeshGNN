# MeshGNN

A C++17 library implementing Graph Neural Networks (GNNs) for 3D mesh simplification. MeshGNN provides tools for processing triangular meshes, extracting geometric features, and applying learning-based edge collapse algorithms for mesh decimation.

## Table of Contents

- [Overview](#overview)
- [Theoretical Background](#theoretical-background)
  - [Graph Neural Networks](#graph-neural-networks)
  - [Mesh Representation as Graphs](#mesh-representation-as-graphs)
  - [Feature Extraction](#feature-extraction)
  - [Mesh Simplification](#mesh-simplification)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Building](#building)
- [Usage](#usage)
- [API Reference](#api-reference)
- [License](#license)

## Overview

MeshGNN bridges the gap between geometric processing and deep learning by treating 3D meshes as graphs and applying GNN architectures to learn optimal mesh simplification strategies. The library is self-contained with no external dependencies beyond the C++17 standard library.

**Key Features:**
- OBJ file loading and mesh representation
- Geometric feature extraction (curvature, normals)
- Graph Convolutional Network (GCN) and EdgeConv layers
- SGD and Adam optimizers with L2 regularization
- Incremental edge collapse mesh simplification
- Model serialization (save/load)

## Theoretical Background

### Graph Neural Networks

Graph Neural Networks extend neural networks to graph-structured data by learning node representations through iterative message passing between neighbors.

#### Graph Convolutional Network (GCN)

The GCN layer implements spectral graph convolutions approximated in the spatial domain:

```
H^(l+1) = σ(Ã · H^(l) · W^(l))
```

Where:
- `H^(l)` is the node feature matrix at layer `l` (N × F_in)
- `Ã` is the normalized adjacency matrix (N × N)
- `W^(l)` is the learnable weight matrix (F_in × F_out)
- `σ` is a non-linear activation function (Mish, ReLU, GELU, or Sigmoid)

The adjacency matrix aggregates features from neighboring nodes, enabling information propagation across the graph structure.

#### EdgeConv Layer

The EdgeConv layer learns edge features by computing differences between connected node features:

```
h_i^(l+1) = Σ_{j∈N(i)} σ((h_i^(l) - h_j^(l)) · W^(l) + b^(l))
```

Where:
- `h_i^(l)` is the feature vector of node `i` at layer `l`
- `N(i)` is the set of neighbors of node `i`
- The difference `(h_i - h_j)` captures local geometric structure

EdgeConv is particularly effective for point cloud and mesh processing as it explicitly models local geometric relationships.

### Mesh Representation as Graphs

A triangular mesh M = (V, E, F) is converted to a graph G = (V, E) where:
- **Vertices V**: Mesh vertices become graph nodes
- **Edges E**: Mesh edges define graph connectivity
- **Adjacency Matrix A**: Sparse matrix where A[i,j] = 1 if vertices i and j share an edge

The adjacency matrix is stored in Compressed Sparse Row (CSR) format for efficient sparse matrix-vector multiplication:
- `vals[]`: Non-zero values
- `col_ind[]`: Column indices
- `row_ptr[]`: Row pointers (row_ptr[i] to row_ptr[i+1] spans row i)

### Feature Extraction

MeshGNN extracts geometric features to provide rich input for the GNN:

#### Node Features (7-dimensional)
1. **Position** (x, y, z): Vertex coordinates
2. **Normal** (nx, ny, nz): Per-vertex normal vectors computed by averaging incident face normals
3. **Gaussian Curvature** (κ): Discrete curvature approximation

**Gaussian Curvature Computation:**

For a vertex v with incident angles θ_i in surrounding triangles:

```
κ(v) = (2π - Σθ_i) / A(v)
```

Where A(v) is the local area (sum of 1/3 of each incident triangle's area). This is the discrete Gauss-Bonnet theorem.

#### Edge Features (2-dimensional)
1. **Edge Length**: Euclidean distance between endpoints
2. **Normal Angle**: Angle between normals of adjacent vertices

### Mesh Simplification

MeshGNN implements edge collapse-based mesh simplification with two strategies:

#### Quadric Error Metric

The cost of collapsing an edge (u, v) is approximated by the edge length:

```
cost(u, v) = ||p_u - p_v||_2
```

This serves as a proxy for the full quadric error metric (QEM), which measures the sum of squared distances to the original surface planes.

#### GNN-Driven Collapse

The GNN learns to predict edge collapse priority scores:

1. Extract node and edge features from the mesh
2. Process features through GNN layers
3. Use output features to score edges for collapse
4. Collapse edges in priority order (lowest cost first)

The simplification algorithm uses **lazy deletion** with versioning for O(E log E) complexity instead of rebuilding the priority queue after each collapse.

#### Edge Collapse Operation

When collapsing edge (u, v):
1. Remove vertex u
2. Move vertex v to the midpoint: `v_new = (u + v) / 2`
3. Update all edges referencing u to reference v
4. Remove degenerate faces (where two or more vertices coincide)
5. Update edge costs for affected edges only

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MeshGNN Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│  │ OBJ File │───▶│ Mesh Object │───▶│ Feature Extraction   │   │
│  └──────────┘    └─────────────┘    │ - Node features (7D) │   │
│                                      │ - Edge features (2D) │   │
│                                      └──────────┬───────────┘   │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    GNN Pipeline                           │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐              │   │
│  │  │  GCN    │───▶│  GCN    │───▶│EdgeConv │───▶ Output   │   │
│  │  │ Layer 1 │    │ Layer 2 │    │  Layer  │    Features  │   │
│  │  └─────────┘    └─────────┘    └─────────┘              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Mesh Simplification                      │   │
│  │  - Edge collapse based on GNN scores                      │   │
│  │  - Priority queue with lazy deletion                      │   │
│  │  - Incremental updates                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
MeshGNN/
├── include/gnnmath/           # Header files
│   ├── types.hpp              # Type aliases (scalar_t, index_t)
│   ├── vector.hpp             # Vector operations and activations
│   ├── matrix.hpp             # Dense and sparse matrix operations
│   ├── mesh.hpp               # Mesh representation
│   ├── graph.hpp              # Graph data structure
│   ├── feature_extraction.hpp # Geometric feature computation
│   ├── mesh_processor.hpp     # Mesh simplification algorithms
│   ├── random.hpp             # Random number utilities
│   └── gnn/
│       ├── layer.hpp          # GCN and EdgeConv layers
│       ├── pipeline.hpp       # Layer stacking and model I/O
│       └── training.hpp       # Trainer with optimizers and losses
│
├── src/                       # Implementation files
│   ├── vector.cpp
│   ├── matrix.cpp
│   ├── mesh.cpp
│   ├── graph.cpp
│   ├── feature_extraction.cpp
│   ├── mesh_processor.cpp
│   ├── random.cpp
│   └── gnn/
│       ├── layer.cpp
│       ├── pipeline.cpp
│       └── training.cpp
│
├── tests/                     # Unit tests (Google Test)
│   ├── test_vector.cpp
│   ├── test_matrix.cpp
│   ├── test_mesh.cpp
│   └── test_gnn.cpp
│
├── data/                      # Sample mesh files
│   └── Porsche_911_GT2.obj
│
├── main.cpp                   # Demo application
├── CMakeLists.txt             # Build configuration
├── CLAUDE.md                  # Claude Code guidance
└── README.md                  # This file
```

## Building

### Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+ (optional, for build system)
- Google Test (optional, for unit tests)
- Intel TBB (optional, for parallel execution)

### CMake Build (Recommended)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build with Tests

```bash
cmake .. -DMESHGNN_BUILD_TESTS=ON
make
ctest --output-on-failure
```

### Manual Compilation

```bash
g++ -std=c++17 -O2 -I include main.cpp src/*.cpp src/gnn/*.cpp -o meshgnn
```

## Usage

### Basic Example

```cpp
#include <gnnmath/mesh.hpp>
#include <gnnmath/feature_extraction.hpp>
#include <gnnmath/gnn/pipeline.hpp>
#include <gnnmath/gnn/layer.hpp>
#include <gnnmath/gnn/training.hpp>

using namespace gnnmath;

int main() {
    // Load mesh
    mesh::mesh m;
    m.load_obj("model.obj");

    // Extract features
    auto node_features = mesh::compute_combined_node_features(m);
    auto adj = m.to_adjacency_matrix();

    // Create GNN pipeline
    gnn::pipeline pipeline;
    pipeline.add_layer(std::make_unique<gnn::gcn_layer>(7, 16));
    pipeline.add_layer(std::make_unique<gnn::gcn_layer>(16, 8));
    pipeline.add_layer(std::make_unique<gnn::edge_conv_layer>(8, 1));

    // Process through GNN
    auto output = pipeline.process(node_features, adj);

    // Save trained model
    pipeline.save("model.bin");

    return 0;
}
```

### Training Example

```cpp
// Create trainer with Adam optimizer and weight decay
gnn::trainer trainer(&pipeline, 0.001, gnn::optimizer_type::ADAM, 0.0001);

// Training loop
for (int epoch = 0; epoch < 100; ++epoch) {
    trainer.train_step(features, adj, targets);

    auto predictions = pipeline.process(features, adj);
    double loss = trainer.mse_loss(predictions, targets);
    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
}
```

### Mesh Simplification

```cpp
// Simplify to 50% of original vertices using GNN scores
std::vector<double> gnn_scores = /* from GNN output */;
mesh::simplify_gnn_edge_collapse(m, m.n_vertices() / 2, gnn_scores);

// Or use basic random removal
mesh::simplify_random_removal(m, target_vertices);
```

## API Reference

### Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| ReLU | max(0, x) | General purpose, sparse activations |
| Sigmoid | 1/(1+e^(-x)) | Output layer for binary classification |
| Mish | x·tanh(softplus(x)) | Smooth, self-regularized (default) |
| GELU | x·Φ(x) | Transformer-style networks |
| Softmax | e^(x_i)/Σe^(x_j) | Multi-class classification output |

### Loss Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| MSE | Σ(y-ŷ)²/N | Regression tasks |
| Cross-Entropy | -Σy·log(ŷ) | Classification tasks |

### Optimizers

| Optimizer | Features |
|-----------|----------|
| SGD | Basic gradient descent with optional weight decay |
| Adam | Adaptive learning rates with momentum (β₁=0.9, β₂=0.999) |

## License

MIT License - see [LICENSE](LICENSE) file for details.

