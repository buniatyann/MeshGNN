# Graph Module

The graph module provides the graph data structure that serves as the interface between mesh geometry and GNN processing. It encapsulates node features, edge features, and connectivity information in a format suitable for graph neural network operations.

## Files

- `include/gnnmath/graph.hpp` / `src/graph.cpp` - Graph structure and operations

---

## Graph Structure

### Class Definition

```cpp
namespace gnnmath {
    struct graph {
        // Graph size
        index_t num_vertices;

        // Connectivity
        std::vector<std::pair<index_t, index_t>> edges;

        // Features
        feature_matrix_t node_features;  // [num_vertices × node_feature_dim]
        feature_matrix_t edge_features;  // [num_edges × edge_feature_dim]

        // Adjacency list: vertex → [(neighbor, edge_index), ...]
        std::map<index_t, std::vector<std::pair<index_t, index_t>>> adjacency;

        // Construction
        graph() = default;
        graph(index_t num_vertices,
              const std::vector<std::pair<index_t, index_t>>& edges,
              const feature_matrix_t& node_features,
              const feature_matrix_t& edge_features);

        // Factory methods
        static graph from_mesh(const mesh& m);

        // Operations
        bool validate() const;
        sparse_matrix to_adjacency_matrix() const;
        void message_passing(/* ... */);
        feature_matrix_t aggregate_features(aggregation_type type) const;
        void update_features(const feature_matrix_t& new_node_features,
                            const feature_matrix_t& new_edge_features);

        // Queries
        const std::vector<std::pair<index_t, index_t>>& get_neighbors(index_t v) const;
        std::vector<index_t> compute_degree() const;
        sparse_matrix laplacian_matrix() const;
    };
}
```

### Data Layout

```
Graph with 4 vertices and 4 edges:

    0 ─────── 1
    │ ╲     ╱ │
    │   ╲ ╱   │
    │   ╱ ╲   │
    │ ╱     ╲ │
    3 ─────── 2

num_vertices = 4

edges = [(0,1), (0,3), (1,2), (2,3), (0,2), (1,3)]
        (undirected, stored once per edge)

node_features = [
    [x₀, y₀, z₀, nx₀, ny₀, nz₀, κ₀],  // vertex 0
    [x₁, y₁, z₁, nx₁, ny₁, nz₁, κ₁],  // vertex 1
    [x₂, y₂, z₂, nx₂, ny₂, nz₂, κ₂],  // vertex 2
    [x₃, y₃, z₃, nx₃, ny₃, nz₃, κ₃],  // vertex 3
]

edge_features = [
    [len₀, θ₀],  // edge (0,1)
    [len₁, θ₁],  // edge (0,3)
    ...
]

adjacency = {
    0: [(1, 0), (3, 1), (2, 4)],  // vertex 0's neighbors with edge indices
    1: [(0, 0), (2, 2), (3, 5)],
    2: [(1, 2), (3, 3), (0, 4)],
    3: [(0, 1), (2, 3), (1, 5)],
}
```

---

## Construction

### Direct Construction

```cpp
graph(index_t num_vertices,
      const std::vector<std::pair<index_t, index_t>>& edges,
      const feature_matrix_t& node_features,
      const feature_matrix_t& edge_features);
```

**Validation performed:**
- `node_features.size() == num_vertices`
- `edge_features.size() == edges.size()`
- All edge vertex indices are < `num_vertices`
- All feature vectors have consistent dimensions

**Builds:**
- Bidirectional adjacency list from edge list
- Edge index references for each adjacency entry

**Example:**
```cpp
index_t num_v = 3;
std::vector<std::pair<index_t, index_t>> edges = {{0, 1}, {1, 2}, {0, 2}};

feature_matrix_t node_feat = {
    {1.0, 0.0, 0.0},  // vertex 0
    {0.0, 1.0, 0.0},  // vertex 1
    {0.0, 0.0, 1.0},  // vertex 2
};

feature_matrix_t edge_feat = {
    {1.0},  // edge 0-1
    {1.0},  // edge 1-2
    {1.414},  // edge 0-2
};

gnnmath::graph g(num_v, edges, node_feat, edge_feat);
```

### From Mesh

```cpp
static graph from_mesh(const mesh& m);
```

Creates a graph from a mesh with computed geometric features.

**Process:**
1. Extract vertices, edges from mesh
2. Compute 7-dimensional node features (position + normal + curvature)
3. Compute 2-dimensional edge features (length + normal angle)
4. Build adjacency structure

**Example:**
```cpp
gnnmath::mesh m;
m.load_obj("model.obj");

auto g = gnnmath::graph::from_mesh(m);

std::cout << "Nodes: " << g.num_vertices << "\n";
std::cout << "Edges: " << g.edges.size() << "\n";
std::cout << "Node feature dim: " << g.node_features[0].size() << "\n";  // 7
std::cout << "Edge feature dim: " << g.edge_features[0].size() << "\n";  // 2
```

---

## Validation

### `validate()`

Checks graph consistency and data integrity.

```cpp
bool validate() const;
```

**Checks:**
1. `node_features.size() == num_vertices`
2. `edge_features.size() == edges.size()`
3. All node feature vectors have same dimension
4. All edge feature vectors have same dimension
5. All edge vertex indices are valid (< num_vertices)
6. Adjacency list is consistent with edge list
7. All feature values are finite (no NaN/Inf)

**Returns:** `true` if all checks pass

**Example:**
```cpp
if (!g.validate()) {
    throw std::runtime_error("Invalid graph structure");
}
```

---

## Adjacency Matrix Conversion

### `to_adjacency_matrix()`

Converts graph connectivity to sparse CSR adjacency matrix.

```cpp
sparse_matrix to_adjacency_matrix() const;
```

**Returns:** Sparse matrix A where:
- A[i][j] = 1 if edge (i, j) exists
- A[i][j] = 0 otherwise
- Matrix is symmetric for undirected graphs

**Implementation:**
```cpp
// Build COO format first, then convert to CSR
std::vector<std::tuple<index_t, index_t, scalar_t>> triplets;
for (const auto& [u, v] : edges) {
    triplets.push_back({u, v, 1.0});
    triplets.push_back({v, u, 1.0});  // Symmetric
}
return sparse_matrix::from_triplets(num_vertices, num_vertices, triplets);
```

**Usage:**
```cpp
auto adj = g.to_adjacency_matrix();

// Use in GNN layer
auto output = gcn_layer.forward(g.node_features, adj);
```

---

## Message Passing

### Concept

Message passing is the core operation in Graph Neural Networks. Each node aggregates information from its neighbors to update its representation.

```
For each node v:
    messages = []
    for each neighbor u:
        message = MESSAGE(h_u, h_v, e_uv)
        messages.append(message)

    aggregated = AGGREGATE(messages)
    h_v_new = UPDATE(h_v, aggregated)
```

### `message_passing()`

Performs one round of message passing.

```cpp
void message_passing(
    std::function<feature_t(const feature_t&, const feature_t&, const feature_t&)> message_fn,
    std::function<feature_t(const std::vector<feature_t>&)> aggregate_fn,
    std::function<feature_t(const feature_t&, const feature_t&)> update_fn
);
```

**Parameters:**
- `message_fn(h_u, h_v, e_uv)`: Computes message from u to v
- `aggregate_fn(messages)`: Aggregates all incoming messages
- `update_fn(h_v, aggregated)`: Updates node v's features

**Example:**
```cpp
// Simple sum aggregation with linear update
g.message_passing(
    // Message: just send source features
    [](const feature_t& h_u, const feature_t& h_v, const feature_t& e_uv) {
        return h_u;
    },
    // Aggregate: sum
    [](const std::vector<feature_t>& msgs) {
        feature_t sum(msgs[0].size(), 0.0);
        for (const auto& m : msgs) {
            for (size_t i = 0; i < sum.size(); ++i) {
                sum[i] += m[i];
            }
        }
        return sum;
    },
    // Update: concatenate and pass through
    [](const feature_t& h_v, const feature_t& agg) {
        feature_t updated = h_v;
        for (scalar_t val : agg) {
            updated.push_back(val);
        }
        return updated;
    }
);
```

---

## Feature Aggregation

### `aggregate_features(type)`

Aggregates neighbor features using specified method.

```cpp
enum class aggregation_type {
    SUM,
    MEAN,
    MAX
};

feature_matrix_t aggregate_features(aggregation_type type) const;
```

**Returns:** Matrix where each row is the aggregation of that vertex's neighbors' features.

**Aggregation Types:**

| Type | Formula | Use Case |
|------|---------|----------|
| SUM | Σⱼ hⱼ | When neighbor count matters |
| MEAN | (1/|N|) Σⱼ hⱼ | Normalized, degree-invariant |
| MAX | maxⱼ hⱼ (element-wise) | Capture dominant features |

**Example:**
```cpp
// Get mean of neighbor features
auto aggregated = g.aggregate_features(aggregation_type::MEAN);

// aggregated[i] = mean of all neighbors of vertex i
```

---

## Feature Updates

### `update_features()`

Replaces node and/or edge features.

```cpp
void update_features(
    const feature_matrix_t& new_node_features,
    const feature_matrix_t& new_edge_features
);
```

**Validates:**
- `new_node_features.size() == num_vertices`
- `new_edge_features.size() == edges.size()`

**Example:**
```cpp
// After GNN processing, update with learned features
auto new_features = pipeline.process(g.node_features, adj);
g.update_features(new_features, g.edge_features);
```

---

## Topology Queries

### `get_neighbors(v)`

Returns neighbors of vertex v with edge indices.

```cpp
const std::vector<std::pair<index_t, index_t>>& get_neighbors(index_t v) const;
```

**Returns:** Vector of (neighbor_vertex, edge_index) pairs

**Example:**
```cpp
for (const auto& [neighbor, edge_idx] : g.get_neighbors(0)) {
    std::cout << "Neighbor " << neighbor
              << " via edge " << edge_idx
              << " (length: " << g.edge_features[edge_idx][0] << ")\n";
}
```

### `compute_degree()`

Computes degree (neighbor count) for each vertex.

```cpp
std::vector<index_t> compute_degree() const;
```

**Returns:** Vector where `degrees[i]` = number of neighbors of vertex i

**Example:**
```cpp
auto degrees = g.compute_degree();
index_t max_degree = *std::max_element(degrees.begin(), degrees.end());
std::cout << "Max degree: " << max_degree << "\n";
```

---

## Graph Laplacian

### `laplacian_matrix()`

Computes the graph Laplacian matrix.

```cpp
sparse_matrix laplacian_matrix() const;
```

**Formula:** L = D - A

Where:
- D = diagonal degree matrix
- A = adjacency matrix

**Properties:**
- Symmetric positive semi-definite
- L × 1 = 0 (constant vector in nullspace)
- Eigenvalues encode connectivity

**Example:**
```cpp
auto L = g.laplacian_matrix();

// Laplacian has property: L * all_ones = zeros
std::vector<scalar_t> ones(g.num_vertices, 1.0);
auto result = L.multiply(ones);  // Should be all zeros
```

---

## Integration with GNN Pipeline

The graph module serves as the bridge between geometry and neural networks:

```
┌─────────────────────────────────────────────────────────┐
│                         Mesh                            │
│  vertices, faces, normals, topology                     │
└─────────────────────────────────────────────────────────┘
                           │
                           │ graph::from_mesh()
                           ▼
┌─────────────────────────────────────────────────────────┐
│                        Graph                            │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ num_vertices │  │    edges     │  │  adjacency   │  │
│  │     = N      │  │ [(u,v), ...] │  │  {v: [...]}  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────────────┐  ┌───────────────────────┐   │
│  │   node_features      │  │    edge_features      │   │
│  │   [N × 7]            │  │    [E × 2]            │   │
│  │   pos + normal + κ   │  │    length + angle     │   │
│  └──────────────────────┘  └───────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                           │
                           │ to_adjacency_matrix()
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Sparse Adjacency                      │
│                     [N × N] CSR                         │
└─────────────────────────────────────────────────────────┘
                           │
                           │ GNN forward pass
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    GNN Pipeline                         │
│                                                         │
│   node_features ──▶ GCN Layer ──▶ ... ──▶ Output       │
│                        ▲                                │
│                        │                                │
│                   adjacency                             │
└─────────────────────────────────────────────────────────┘
```

### Complete Example

```cpp
#include <gnnmath/gnnmath.hpp>

int main() {
    // Load mesh
    gnnmath::mesh m;
    m.load_obj("bunny.obj");

    // Create graph with geometric features
    auto g = gnnmath::graph::from_mesh(m);

    // Validate structure
    if (!g.validate()) {
        throw std::runtime_error("Invalid graph");
    }

    // Print statistics
    std::cout << "Graph statistics:\n";
    std::cout << "  Vertices: " << g.num_vertices << "\n";
    std::cout << "  Edges: " << g.edges.size() << "\n";
    std::cout << "  Node feature dim: " << g.node_features[0].size() << "\n";
    std::cout << "  Edge feature dim: " << g.edge_features[0].size() << "\n";

    auto degrees = g.compute_degree();
    scalar_t avg_degree = 0;
    for (auto d : degrees) avg_degree += d;
    avg_degree /= g.num_vertices;
    std::cout << "  Avg degree: " << avg_degree << "\n";

    // Convert to sparse adjacency
    auto adj = g.to_adjacency_matrix();
    std::cout << "  Adjacency nnz: " << adj.nnz() << "\n";

    // Build GNN pipeline
    auto pipeline = std::make_shared<gnnmath::gnn::pipeline>();
    pipeline->add_layer(std::make_unique<gnnmath::gnn::gcn_layer>(7, 32));
    pipeline->add_layer(std::make_unique<gnnmath::gnn::gcn_layer>(32, 16));
    pipeline->add_layer(std::make_unique<gnnmath::gnn::gcn_layer>(16, 1));

    // Forward pass
    auto output = pipeline->process(g.node_features, adj);

    // Output contains per-vertex scores for mesh simplification
    std::cout << "Output feature dim: " << output[0].size() << "\n";

    return 0;
}
```

---

## Performance Considerations

### Memory Usage

| Component | Memory |
|-----------|--------|
| node_features | O(V × d_node) |
| edge_features | O(E × d_edge) |
| edges | O(E) |
| adjacency | O(V + E) |

For a mesh with V=100k vertices and E=300k edges:
- 7-dim node features: ~5.6 MB
- 2-dim edge features: ~4.8 MB
- Adjacency: ~3.6 MB

### Access Patterns

- **Neighbor iteration**: O(degree) per vertex via adjacency map
- **Edge lookup**: O(log V) via adjacency map
- **Feature access**: O(1) via vector indexing

### Optimization Tips

1. **Pre-allocate features**: Reserve vector capacity before populating
2. **Batch processing**: Process multiple graphs together when possible
3. **Cache adjacency matrix**: Recompute only when topology changes
4. **Use move semantics**: Avoid copying large feature matrices

```cpp
// Good: move features into graph
feature_matrix_t features = compute_features();
graph g(num_v, edges, std::move(features), edge_features);

// Bad: copy features
graph g(num_v, edges, features, edge_features);  // Copies!
```
