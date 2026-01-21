# API Reference

Complete API reference for all MeshGNN classes and functions.

---

## Table of Contents

1. [Core Types](#core-types)
2. [Configuration](#configuration)
3. [Random](#random)
4. [Vector Operations](#vector-operations)
5. [Dense Matrix](#dense-matrix)
6. [Sparse Matrix](#sparse-matrix)
7. [OBJ Loader](#obj-loader)
8. [Mesh](#mesh)
9. [Feature Extraction](#feature-extraction)
10. [Mesh Processor](#mesh-processor)
11. [Graph](#graph)
12. [GNN Layers](#gnn-layers)
13. [Pipeline](#pipeline)
14. [Optimizers](#optimizers)
15. [Loss Functions](#loss-functions)
16. [Trainer](#trainer)

---

## Core Types

**Header:** `include/gnnmath/core/types.hpp`

```cpp
namespace gnnmath {
    using scalar_t = double;
    using index_t = std::size_t;
    using feature_t = std::vector<scalar_t>;
    using feature_matrix_t = std::vector<feature_t>;
}
```

| Type | Definition | Description |
|------|------------|-------------|
| `scalar_t` | `double` | Floating-point type |
| `index_t` | `std::size_t` | Index/size type |
| `feature_t` | `std::vector<scalar_t>` | Single feature vector |
| `feature_matrix_t` | `std::vector<feature_t>` | Matrix of features |

---

## Configuration

**Header:** `include/gnnmath/core/config.hpp`

```cpp
namespace gnnmath::config {
    constexpr int version_major;
    constexpr int version_minor;
    constexpr int version_patch;

    constexpr scalar_t epsilon;           // 1e-10
    constexpr scalar_t exp_max;           // 700.0
    constexpr index_t parallel_threshold; // 1000

    constexpr scalar_t adam_beta1;        // 0.9
    constexpr scalar_t adam_beta2;        // 0.999
    constexpr scalar_t adam_epsilon;      // 1e-8
}
```

---

## Random

**Header:** `include/gnnmath/core/random.hpp`

### Functions

```cpp
namespace gnnmath {
    // Generate uniform random scalar in [min, max]
    scalar_t uniform(scalar_t min, scalar_t max);

    // Generate vector of n uniform random scalars
    std::vector<scalar_t> uniform_vector(index_t n, scalar_t min, scalar_t max);

    // Set random seed for reproducibility
    void seed(unsigned int s);
}
```

---

## Vector Operations

**Header:** `include/gnnmath/math/vector.hpp`

### Arithmetic

```cpp
namespace gnnmath::vector {
    // Element-wise addition
    std::vector<scalar_t> operator+(const std::vector<scalar_t>& a,
                                    const std::vector<scalar_t>& b);

    // Element-wise subtraction
    std::vector<scalar_t> operator-(const std::vector<scalar_t>& a,
                                    const std::vector<scalar_t>& b);

    // In-place addition
    std::vector<scalar_t>& operator+=(std::vector<scalar_t>& a,
                                      const std::vector<scalar_t>& b);

    // In-place subtraction
    std::vector<scalar_t>& operator-=(std::vector<scalar_t>& a,
                                      const std::vector<scalar_t>& b);

    // Scalar multiplication
    std::vector<scalar_t> scalar_multiply(const std::vector<scalar_t>& a,
                                          scalar_t b);

    // Dot product
    scalar_t dot_product(const std::vector<scalar_t>& a,
                         const std::vector<scalar_t>& b);

    // Euclidean norm (L2)
    scalar_t euclidean_norm(const std::vector<scalar_t>& a);
}
```

### Activation Functions

```cpp
namespace gnnmath::vector {
    std::vector<scalar_t> relu(const std::vector<scalar_t>& a);
    std::vector<scalar_t> sigmoid(const std::vector<scalar_t>& a);
    std::vector<scalar_t> mish(const std::vector<scalar_t>& a);
    std::vector<scalar_t> softmax(const std::vector<scalar_t>& a);
    std::vector<scalar_t> softplus(const std::vector<scalar_t>& a);
    std::vector<scalar_t> gelu(const std::vector<scalar_t>& a);
    std::vector<scalar_t> silu(const std::vector<scalar_t>& a);
    std::vector<scalar_t> softsign(const std::vector<scalar_t>& a);
}
```

---

## Dense Matrix

**Header:** `include/gnnmath/math/dense_matrix.hpp`

### Class: `dense_matrix`

```cpp
namespace gnnmath::matrix {
    class dense_matrix {
    public:
        // Constructors
        dense_matrix(index_t rows, index_t cols);
        dense_matrix(index_t rows, index_t cols,
                     const std::vector<scalar_t>& data);

        // Element access
        scalar_t& operator()(index_t i, index_t j);
        scalar_t operator()(index_t i, index_t j) const;

        // Dimensions
        index_t rows() const;
        index_t cols() const;

        // Raw data
        const std::vector<scalar_t>& data() const;
        std::vector<scalar_t>& data();
    };
}
```

### Free Functions

```cpp
namespace gnnmath::matrix {
    // Matrix-vector multiplication
    std::vector<scalar_t> matrix_vector_multiply(const dense_matrix& A,
                                                 const std::vector<scalar_t>& v);

    // Matrix multiplication
    dense_matrix operator*(const dense_matrix& A, const dense_matrix& B);

    // Transpose
    dense_matrix transpose(const dense_matrix& A);

    // Element-wise operations
    dense_matrix operator+(const dense_matrix& A, const dense_matrix& B);
    dense_matrix operator-(const dense_matrix& A, const dense_matrix& B);
    dense_matrix elementwise_multiply(const dense_matrix& A, const dense_matrix& B);

    // Utilities
    dense_matrix I(index_t n);                           // Identity matrix
    scalar_t frobenius_norm(const dense_matrix& A);      // ||A||_F
    std::vector<scalar_t> extract_diagonal(const dense_matrix& A);
    bool is_valid(const dense_matrix& A);                // Check for NaN/Inf
}
```

---

## Sparse Matrix

**Header:** `include/gnnmath/math/sparse_matrix.hpp`

### Class: `sparse_matrix`

```cpp
namespace gnnmath::matrix {
    class sparse_matrix {
    public:
        // Constructors
        sparse_matrix(index_t rows, index_t cols);       // Empty
        sparse_matrix(const dense_matrix& dense);        // From dense
        sparse_matrix(index_t rows, index_t cols,
                      std::vector<scalar_t>&& vals,
                      std::vector<index_t>&& col_ind,
                      std::vector<index_t>&& row_ptr);   // From CSR arrays

        // Dimensions
        index_t rows() const;
        index_t cols() const;
        index_t nnz() const;

        // CSR data access
        const std::vector<scalar_t>& vals() const;
        const std::vector<index_t>& col_ind() const;
        const std::vector<index_t>& row_ptr() const;

        // Operations
        std::vector<scalar_t> multiply(const std::vector<scalar_t>& x) const;
        bool validate() const;
    };
}
```

### Free Functions

```cpp
namespace gnnmath::matrix {
    // Arithmetic
    sparse_matrix operator+(const sparse_matrix& A, const sparse_matrix& B);
    sparse_matrix operator-(const sparse_matrix& A, const sparse_matrix& B);
    sparse_matrix sparse_matrix_multiply(const sparse_matrix& A,
                                         const sparse_matrix& B);
    sparse_matrix sparse_transpose(const sparse_matrix& A);

    // Construction
    sparse_matrix Identity(index_t n);
    sparse_matrix build_adj_matrix(index_t num_vertices,
                                   const std::vector<std::pair<index_t, index_t>>& edges);

    // Graph operations
    std::vector<scalar_t> compute_degrees(const sparse_matrix& A);
    sparse_matrix laplacian_matrix(const sparse_matrix& A);
    sparse_matrix normalized_laplacian_matrix(const sparse_matrix& A);

    // Utilities
    bool is_symmetric(const sparse_matrix& A);
    dense_matrix to_dense(const sparse_matrix& A);
}
```

---

## OBJ Loader

**Header:** `include/gnnmath/geometry/obj_loader.hpp`

### Structures

```cpp
namespace gnnmath {
    enum class triangulation_method { FAN, EAR_CLIPPING, DELAUNAY };

    struct load_options {
        bool triangulate = true;
        triangulation_method tri_method = triangulation_method::FAN;
        bool generate_normals = true;
        bool flip_normals = false;
        bool flip_texcoords_v = false;
        scalar_t scale = 1.0;
        std::array<scalar_t, 3> offset = {0, 0, 0};
        bool strict_mode = false;
    };

    struct material {
        std::string name;
        std::array<scalar_t, 3> ambient;
        std::array<scalar_t, 3> diffuse;
        std::array<scalar_t, 3> specular;
        scalar_t shininess;
        scalar_t opacity;
        std::string diffuse_map;
    };

    struct group {
        std::string name;
        index_t face_start;
        index_t face_count;
        std::string material_name;
    };

    struct obj_data {
        std::vector<std::array<scalar_t, 3>> vertices;
        std::vector<std::array<scalar_t, 2>> texcoords;
        std::vector<std::array<scalar_t, 3>> normals;
        std::vector<std::vector<std::array<index_t, 3>>> faces;
        std::vector<material> materials;
        std::map<std::string, index_t> material_map;
        std::vector<group> groups;
        std::vector<std::string> warnings;
        std::string filename;
    };
}
```

### Functions

```cpp
namespace gnnmath {
    obj_data load_obj(const std::string& filename,
                      const load_options& options = load_options{});
}
```

---

## Mesh

**Header:** `include/gnnmath/geometry/mesh.hpp`

### Class: `mesh`

```cpp
namespace gnnmath {
    class mesh {
    public:
        // Construction
        mesh() = default;
        void load_obj(const std::string& filename);

        // Geometry access
        const std::vector<std::array<scalar_t, 3>>& vertices() const;
        const std::vector<std::pair<index_t, index_t>>& edges() const;
        const std::vector<std::array<index_t, 3>>& faces() const;
        const std::vector<std::array<scalar_t, 2>>& texcoords() const;
        const std::vector<std::array<scalar_t, 3>>& file_normals() const;

        // Counts
        index_t num_vertices() const;
        index_t num_edges() const;
        index_t num_faces() const;

        // Topology queries
        const std::vector<index_t>& get_neighbors(index_t v) const;
        const std::vector<index_t>& get_incident_edges(index_t v) const;
        index_t get_edge_index(index_t u, index_t v) const;

        // Feature computation
        feature_matrix_t compute_node_features() const;
        feature_matrix_t compute_edge_features() const;
        std::vector<std::array<scalar_t, 3>> compute_normals() const;

        // Conversion
        sparse_matrix to_adjacency_matrix() const;

        // Utilities
        bool is_valid() const;
        std::vector<index_t> sample_vertices(index_t n) const;
        void add_vertex_noise(scalar_t scale);
    };
}
```

---

## Feature Extraction

**Header:** `include/gnnmath/geometry/feature_extraction.hpp`

### Functions

```cpp
namespace gnnmath {
    // Gaussian curvature per vertex
    std::vector<scalar_t> compute_gaussian_curvature(const mesh& m);

    // Node features [N × 7]: position (3) + normal (3) + curvature (1)
    feature_matrix_t compute_node_features(const mesh& m);

    // Edge features [E × 2]: length (1) + normal angle (1)
    feature_matrix_t compute_edge_features(const mesh& m);
}
```

---

## Mesh Processor

**Header:** `include/gnnmath/geometry/mesh_processor.hpp`

### Functions

```cpp
namespace gnnmath {
    // Quadric error for edge collapse
    scalar_t compute_quadric_error(const mesh& m, index_t u, index_t v);

    // GNN-driven simplification
    mesh simplify_with_gnn_scores(mesh& m,
                                  index_t target_vertices,
                                  const std::vector<scalar_t>& scores);

    // Random baseline simplification
    mesh simplify_random_removal(mesh& m, index_t target_vertices);
}
```

---

## Graph

**Header:** `include/gnnmath/graph.hpp`

### Structure: `graph`

```cpp
namespace gnnmath {
    enum class aggregation_type { SUM, MEAN, MAX };

    struct graph {
        // Data
        index_t num_vertices;
        std::vector<std::pair<index_t, index_t>> edges;
        feature_matrix_t node_features;
        feature_matrix_t edge_features;
        std::map<index_t, std::vector<std::pair<index_t, index_t>>> adjacency;

        // Construction
        graph() = default;
        graph(index_t num_vertices,
              const std::vector<std::pair<index_t, index_t>>& edges,
              const feature_matrix_t& node_features,
              const feature_matrix_t& edge_features);

        // Factory
        static graph from_mesh(const mesh& m);

        // Operations
        bool validate() const;
        sparse_matrix to_adjacency_matrix() const;
        sparse_matrix laplacian_matrix() const;
        feature_matrix_t aggregate_features(aggregation_type type) const;
        void update_features(const feature_matrix_t& new_node_features,
                            const feature_matrix_t& new_edge_features);

        // Queries
        const std::vector<std::pair<index_t, index_t>>& get_neighbors(index_t v) const;
        std::vector<index_t> compute_degree() const;
    };
}
```

---

## GNN Layers

**Header:** `include/gnnmath/gnn/layers/layer.hpp`

### Enums

```cpp
namespace gnnmath::gnn {
    enum class activation_type { RELU, MISH, SIGMOID, GELU };
}
```

### Abstract Class: `layer`

```cpp
namespace gnnmath::gnn {
    class layer {
    public:
        virtual ~layer() = default;

        virtual feature_matrix_t forward(const feature_matrix_t& input,
                                        const sparse_matrix& adjacency) = 0;
        virtual index_t in_features() const = 0;
        virtual index_t out_features() const = 0;

        // Public weights for training
        dense_matrix W;
        feature_t b;
    };
}
```

### Class: `gcn_layer`

**Header:** `include/gnnmath/gnn/layers/gcn_layer.hpp`

```cpp
namespace gnnmath::gnn {
    class gcn_layer : public layer {
    public:
        gcn_layer(index_t in_features,
                  index_t out_features,
                  activation_type activation = activation_type::RELU);

        feature_matrix_t forward(const feature_matrix_t& input,
                                const sparse_matrix& adjacency) override;
        index_t in_features() const override;
        index_t out_features() const override;
    };
}
```

### Class: `edge_conv_layer`

**Header:** `include/gnnmath/gnn/layers/edge_conv_layer.hpp`

```cpp
namespace gnnmath::gnn {
    class edge_conv_layer : public layer {
    public:
        edge_conv_layer(index_t in_features,
                        index_t out_features,
                        activation_type activation = activation_type::RELU);

        feature_matrix_t forward(const feature_matrix_t& input,
                                const sparse_matrix& adjacency) override;
        index_t in_features() const override;
        index_t out_features() const override;
    };
}
```

---

## Pipeline

**Header:** `include/gnnmath/gnn/pipeline.hpp`

### Class: `pipeline`

```cpp
namespace gnnmath::gnn {
    class pipeline {
    public:
        pipeline() = default;

        // Layer management
        void add_layer(std::unique_ptr<layer> layer);
        index_t num_layers() const;
        layer& get_layer(index_t i);
        const layer& get_layer(index_t i) const;

        // Forward pass
        feature_matrix_t process(const feature_matrix_t& input,
                                const sparse_matrix& adjacency);
        feature_matrix_t process(const mesh& m);

        // Serialization
        void save(const std::string& filename) const;
        void load(const std::string& filename);
    };
}
```

---

## Optimizers

**Header:** `include/gnnmath/gnn/optimizers/optimizer.hpp`

### Enums

```cpp
namespace gnnmath::gnn {
    enum class optimizer_type { SGD, ADAM };
}
```

### Abstract Class: `optimizer`

```cpp
namespace gnnmath::gnn {
    class optimizer {
    public:
        optimizer(scalar_t learning_rate, scalar_t weight_decay = 0.0);
        virtual ~optimizer() = default;

        virtual void update(dense_matrix& W, const dense_matrix& grad_W,
                           feature_t& b, const feature_t& grad_b,
                           index_t layer_idx) = 0;
        virtual void reset() = 0;
        virtual void prepare_for_layers(index_t num_layers) = 0;

        scalar_t learning_rate() const;
        void set_learning_rate(scalar_t lr);
        scalar_t weight_decay() const;
    };
}
```

### Class: `sgd_optimizer`

**Header:** `include/gnnmath/gnn/optimizers/sgd.hpp`

```cpp
namespace gnnmath::gnn {
    class sgd_optimizer : public optimizer {
    public:
        sgd_optimizer(scalar_t learning_rate, scalar_t weight_decay = 0.0);

        void update(dense_matrix& W, const dense_matrix& grad_W,
                   feature_t& b, const feature_t& grad_b,
                   index_t layer_idx) override;
        void reset() override;
        void prepare_for_layers(index_t num_layers) override;
    };
}
```

### Class: `adam_optimizer`

**Header:** `include/gnnmath/gnn/optimizers/adam.hpp`

```cpp
namespace gnnmath::gnn {
    class adam_optimizer : public optimizer {
    public:
        adam_optimizer(scalar_t learning_rate,
                      scalar_t weight_decay = 0.0,
                      scalar_t beta1 = config::adam_beta1,
                      scalar_t beta2 = config::adam_beta2,
                      scalar_t epsilon = config::adam_epsilon);

        void update(dense_matrix& W, const dense_matrix& grad_W,
                   feature_t& b, const feature_t& grad_b,
                   index_t layer_idx) override;
        void reset() override;
        void prepare_for_layers(index_t num_layers) override;
    };
}
```

---

## Loss Functions

**Header:** `include/gnnmath/gnn/losses/loss.hpp`

### Abstract Class: `loss`

```cpp
namespace gnnmath::gnn {
    class loss {
    public:
        virtual ~loss() = default;

        virtual scalar_t compute(const feature_matrix_t& predictions,
                                const feature_matrix_t& targets) = 0;
        virtual feature_matrix_t gradient(const feature_matrix_t& predictions,
                                         const feature_matrix_t& targets) = 0;
        virtual std::string name() const = 0;
    };
}
```

### Class: `mse_loss`

**Header:** `include/gnnmath/gnn/losses/mse.hpp`

```cpp
namespace gnnmath::gnn {
    class mse_loss : public loss {
    public:
        scalar_t compute(const feature_matrix_t& predictions,
                        const feature_matrix_t& targets) override;
        feature_matrix_t gradient(const feature_matrix_t& predictions,
                                 const feature_matrix_t& targets) override;
        std::string name() const override;  // "MSE"
    };
}
```

### Class: `cross_entropy_loss`

**Header:** `include/gnnmath/gnn/losses/cross_entropy.hpp`

```cpp
namespace gnnmath::gnn {
    class cross_entropy_loss : public loss {
    public:
        scalar_t compute(const feature_matrix_t& predictions,
                        const feature_matrix_t& targets) override;
        feature_matrix_t gradient(const feature_matrix_t& predictions,
                                 const feature_matrix_t& targets) override;
        std::string name() const override;  // "CrossEntropy"
    };
}
```

---

## Trainer

**Header:** `include/gnnmath/gnn/training.hpp`

### Class: `trainer`

```cpp
namespace gnnmath::gnn {
    class trainer {
    public:
        // Constructors
        trainer(std::shared_ptr<pipeline> pipeline,
                optimizer_type opt_type,
                scalar_t learning_rate,
                scalar_t weight_decay = 0.0);

        trainer(std::shared_ptr<pipeline> pipeline,
                std::unique_ptr<optimizer> opt,
                std::unique_ptr<loss> loss_fn);

        // Training
        scalar_t train_step(const feature_matrix_t& input,
                           const sparse_matrix& adjacency,
                           const feature_matrix_t& targets);

        // Evaluation
        scalar_t compute_loss(const feature_matrix_t& input,
                             const sparse_matrix& adjacency,
                             const feature_matrix_t& targets);

        // Convenience loss methods
        scalar_t mse_loss(const feature_matrix_t& predictions,
                         const feature_matrix_t& targets);
        scalar_t cross_entropy_loss(const feature_matrix_t& predictions,
                                   const feature_matrix_t& targets);

        // Configuration
        void set_learning_rate(scalar_t lr);
        scalar_t learning_rate() const;
        void set_weight_decay(scalar_t wd);

        // Access
        optimizer& get_optimizer();
        loss& get_loss();
    };
}
```

---

## Include All

**Header:** `include/gnnmath/gnnmath.hpp`

Master include file that includes all components:

```cpp
#include <gnnmath/core/types.hpp>
#include <gnnmath/core/config.hpp>
#include <gnnmath/core/random.hpp>
#include <gnnmath/math/vector.hpp>
#include <gnnmath/math/dense_matrix.hpp>
#include <gnnmath/math/sparse_matrix.hpp>
#include <gnnmath/geometry/obj_loader.hpp>
#include <gnnmath/geometry/mesh.hpp>
#include <gnnmath/geometry/feature_extraction.hpp>
#include <gnnmath/geometry/mesh_processor.hpp>
#include <gnnmath/graph.hpp>
#include <gnnmath/gnn/layers/layer.hpp>
#include <gnnmath/gnn/layers/gcn_layer.hpp>
#include <gnnmath/gnn/layers/edge_conv_layer.hpp>
#include <gnnmath/gnn/optimizers/optimizer.hpp>
#include <gnnmath/gnn/optimizers/sgd.hpp>
#include <gnnmath/gnn/optimizers/adam.hpp>
#include <gnnmath/gnn/losses/loss.hpp>
#include <gnnmath/gnn/losses/mse.hpp>
#include <gnnmath/gnn/losses/cross_entropy.hpp>
#include <gnnmath/gnn/pipeline.hpp>
#include <gnnmath/gnn/training.hpp>
```

**Usage:**

```cpp
#include <gnnmath/gnnmath.hpp>

int main() {
    gnnmath::mesh m;
    m.load_obj("model.obj");
    // Use all gnnmath components...
}
```
