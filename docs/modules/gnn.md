# GNN Module

The GNN module implements Graph Neural Network components including layers, pipeline management, training infrastructure, optimizers, and loss functions.

## Files

**Layers:**
- `include/gnnmath/gnn/layers/layer.hpp` - Abstract layer interface
- `include/gnnmath/gnn/layers/gcn_layer.hpp` / `src/gnn/layers/gcn_layer.cpp` - Graph Convolutional Network layer
- `include/gnnmath/gnn/layers/edge_conv_layer.hpp` / `src/gnn/layers/edge_conv_layer.cpp` - Edge Convolution layer

**Optimizers:**
- `include/gnnmath/gnn/optimizers/optimizer.hpp` - Abstract optimizer interface
- `include/gnnmath/gnn/optimizers/sgd.hpp` / `src/gnn/optimizers/sgd.cpp` - SGD optimizer
- `include/gnnmath/gnn/optimizers/adam.hpp` / `src/gnn/optimizers/adam.cpp` - Adam optimizer

**Losses:**
- `include/gnnmath/gnn/losses/loss.hpp` - Abstract loss interface
- `include/gnnmath/gnn/losses/mse.hpp` / `src/gnn/losses/mse.cpp` - Mean Squared Error
- `include/gnnmath/gnn/losses/cross_entropy.hpp` / `src/gnn/losses/cross_entropy.cpp` - Cross-Entropy loss

**Pipeline & Training:**
- `include/gnnmath/gnn/pipeline.hpp` / `src/gnn/pipeline.cpp` - Layer stacking and serialization
- `include/gnnmath/gnn/training.hpp` / `src/gnn/training.cpp` - Trainer class

---

## Layer Base Class

### Abstract Interface

```cpp
namespace gnnmath::gnn {
    enum class activation_type {
        RELU,
        MISH,
        SIGMOID,
        GELU
    };

    class layer {
    public:
        virtual ~layer() = default;

        // Forward pass
        virtual feature_matrix_t forward(
            const feature_matrix_t& input,
            const sparse_matrix& adjacency
        ) = 0;

        // Dimensions
        virtual index_t in_features() const = 0;
        virtual index_t out_features() const = 0;

        // Weights accessible for training
        dense_matrix W;  // Weight matrix [in_features × out_features]
        feature_t b;     // Bias vector [out_features]
    };
}
```

### Design Principles

1. **Uniform Interface**: All layers take features + adjacency, return features
2. **Public Weights**: W and b are public for direct trainer access
3. **Activation Configurable**: Each layer can use different activations
4. **Dimension Tracking**: Layers know their input/output dimensions

---

## GCN Layer (Graph Convolutional Network)

### Mathematical Foundation

The GCN layer implements spectral graph convolution approximated in the spatial domain:

```
H^(l+1) = σ(Ã · H^(l) · W^(l) + b^(l))
```

Where:
- H^(l) = node features at layer l [N × d_in]
- Ã = normalized adjacency (or adjacency with self-loops)
- W^(l) = learnable weight matrix [d_in × d_out]
- b^(l) = learnable bias vector [d_out]
- σ = activation function

### Implementation

```cpp
namespace gnnmath::gnn {
    class gcn_layer : public layer {
    private:
        index_t in_dim_;
        index_t out_dim_;
        activation_type activation_;

    public:
        gcn_layer(index_t in_features,
                  index_t out_features,
                  activation_type activation = activation_type::RELU);

        feature_matrix_t forward(
            const feature_matrix_t& input,
            const sparse_matrix& adjacency
        ) override;

        index_t in_features() const override { return in_dim_; }
        index_t out_features() const override { return out_dim_; }
    };
}
```

### Forward Pass Algorithm

```cpp
feature_matrix_t gcn_layer::forward(
    const feature_matrix_t& input,
    const sparse_matrix& adjacency
) {
    index_t num_nodes = input.size();
    feature_matrix_t output(num_nodes);

    for (index_t i = 0; i < num_nodes; ++i) {
        // 1. Aggregate from neighbors using adjacency
        feature_t aggregated(in_dim_, 0.0);
        for (index_t k = adjacency.row_ptr()[i]; k < adjacency.row_ptr()[i+1]; ++k) {
            index_t j = adjacency.col_ind()[k];
            scalar_t weight = adjacency.vals()[k];
            for (index_t d = 0; d < in_dim_; ++d) {
                aggregated[d] += weight * input[j][d];
            }
        }

        // 2. Linear transform: out = aggregated · W + b
        feature_t transformed(out_dim_, 0.0);
        for (index_t o = 0; o < out_dim_; ++o) {
            for (index_t d = 0; d < in_dim_; ++d) {
                transformed[o] += aggregated[d] * W(d, o);
            }
            transformed[o] += b[o];
        }

        // 3. Apply activation
        output[i] = apply_activation(transformed, activation_);
    }

    return output;
}
```

### Weight Initialization

Weights are initialized uniformly in [-0.1, 0.1]:

```cpp
gcn_layer::gcn_layer(index_t in_features, index_t out_features, activation_type activation)
    : in_dim_(in_features), out_dim_(out_features), activation_(activation)
{
    // Xavier-like initialization
    W = dense_matrix(in_features, out_features);
    for (index_t i = 0; i < in_features; ++i) {
        for (index_t j = 0; j < out_features; ++j) {
            W(i, j) = uniform(-0.1, 0.1);
        }
    }

    // Zero bias
    b = feature_t(out_features, 0.0);
}
```

### Example Usage

```cpp
// Create GCN layer: 7 input features → 32 output features
auto gcn = std::make_unique<gnnmath::gnn::gcn_layer>(7, 32, activation_type::RELU);

// Forward pass
auto output = gcn->forward(node_features, adjacency);
// output: [num_nodes × 32]
```

---

## EdgeConv Layer

### Mathematical Foundation

EdgeConv captures local geometric structure by operating on edge differences:

```
h_i^(l+1) = Σ_{j ∈ N(i)} σ((h_i^(l) - h_j^(l)) · W^(l) + b^(l))
```

Where:
- h_i - h_j captures the relative position/features of neighbors
- Aggregation over all neighbors
- Better for point clouds and meshes with local geometric structure

### Implementation

```cpp
namespace gnnmath::gnn {
    class edge_conv_layer : public layer {
    private:
        index_t in_dim_;
        index_t out_dim_;
        activation_type activation_;

    public:
        edge_conv_layer(index_t in_features,
                        index_t out_features,
                        activation_type activation = activation_type::RELU);

        feature_matrix_t forward(
            const feature_matrix_t& input,
            const sparse_matrix& adjacency
        ) override;
    };
}
```

### Forward Pass Algorithm

```cpp
feature_matrix_t edge_conv_layer::forward(
    const feature_matrix_t& input,
    const sparse_matrix& adjacency
) {
    index_t num_nodes = input.size();
    feature_matrix_t output(num_nodes);

    for (index_t i = 0; i < num_nodes; ++i) {
        feature_t aggregated(out_dim_, 0.0);

        // For each neighbor j
        for (index_t k = adjacency.row_ptr()[i]; k < adjacency.row_ptr()[i+1]; ++k) {
            index_t j = adjacency.col_ind()[k];

            // 1. Compute difference: h_i - h_j
            feature_t diff(in_dim_);
            for (index_t d = 0; d < in_dim_; ++d) {
                diff[d] = input[i][d] - input[j][d];
            }

            // 2. Linear transform: diff · W + b
            feature_t edge_out(out_dim_);
            for (index_t o = 0; o < out_dim_; ++o) {
                edge_out[o] = b[o];
                for (index_t d = 0; d < in_dim_; ++d) {
                    edge_out[o] += diff[d] * W(d, o);
                }
            }

            // 3. Apply activation and aggregate
            edge_out = apply_activation(edge_out, activation_);
            for (index_t o = 0; o < out_dim_; ++o) {
                aggregated[o] += edge_out[o];
            }
        }

        output[i] = aggregated;
    }

    return output;
}
```

### GCN vs EdgeConv Comparison

| Aspect | GCN | EdgeConv |
|--------|-----|----------|
| Input | Neighbor features | Feature differences |
| Best for | Node classification | Point clouds, meshes |
| Captures | Global neighborhood | Local geometry |
| Computation | O(E × d²) | O(E × d²) |
| Gradient flow | Through neighbors | Through edges |

---

## GNN Pipeline

The pipeline manages a stack of GNN layers and provides serialization.

### Class Definition

```cpp
namespace gnnmath::gnn {
    class pipeline {
    private:
        std::vector<std::unique_ptr<layer>> layers_;

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

### Adding Layers

```cpp
void pipeline::add_layer(std::unique_ptr<layer> layer) {
    if (!layers_.empty()) {
        // Validate dimension compatibility
        if (layers_.back()->out_features() != layer->in_features()) {
            throw std::invalid_argument(
                "Layer dimension mismatch: " +
                std::to_string(layers_.back()->out_features()) +
                " vs " + std::to_string(layer->in_features())
            );
        }
    }
    layers_.push_back(std::move(layer));
}
```

### Forward Pass

```cpp
feature_matrix_t pipeline::process(
    const feature_matrix_t& input,
    const sparse_matrix& adjacency
) {
    feature_matrix_t current = input;
    for (auto& layer : layers_) {
        current = layer->forward(current, adjacency);
    }
    return current;
}
```

### Serialization Format

Binary format with magic number and version:

```
┌────────────────────────────────────┐
│ Magic Number: 0x4D475050 ("MGPP")  │  4 bytes
├────────────────────────────────────┤
│ Format Version: 1                  │  4 bytes
├────────────────────────────────────┤
│ Number of Layers                   │  4 bytes
├────────────────────────────────────┤
│ Layer 0:                           │
│   - Type (1=GCN, 2=EdgeConv)       │  4 bytes
│   - Activation type                │  4 bytes
│   - Input dimension                │  4 bytes
│   - Output dimension               │  4 bytes
│   - Weight matrix (row-major)      │  in×out × 8 bytes
│   - Bias vector                    │  out × 8 bytes
├────────────────────────────────────┤
│ Layer 1: ...                       │
├────────────────────────────────────┤
│ ...                                │
└────────────────────────────────────┘
```

### Example Pipeline

```cpp
auto pipeline = std::make_shared<gnnmath::gnn::pipeline>();

// Build encoder: 7 → 64 → 32 → 16 → 1
pipeline->add_layer(std::make_unique<gcn_layer>(7, 64, activation_type::RELU));
pipeline->add_layer(std::make_unique<gcn_layer>(64, 32, activation_type::RELU));
pipeline->add_layer(std::make_unique<gcn_layer>(32, 16, activation_type::RELU));
pipeline->add_layer(std::make_unique<gcn_layer>(16, 1, activation_type::SIGMOID));

// Process
auto scores = pipeline->process(features, adjacency);

// Save trained model
pipeline->save("model.bin");

// Load later
auto loaded_pipeline = std::make_shared<gnnmath::gnn::pipeline>();
loaded_pipeline->load("model.bin");
```

---

## Optimizers

### Base Class

```cpp
namespace gnnmath::gnn {
    class optimizer {
    protected:
        scalar_t learning_rate_;
        scalar_t weight_decay_;

    public:
        optimizer(scalar_t lr, scalar_t wd = 0.0)
            : learning_rate_(lr), weight_decay_(wd) {}

        virtual ~optimizer() = default;

        // Apply gradient update to parameters
        virtual void update(dense_matrix& W, const dense_matrix& grad_W,
                           feature_t& b, const feature_t& grad_b,
                           index_t layer_idx) = 0;

        // Reset optimizer state (e.g., momentum)
        virtual void reset() = 0;

        // Prepare state for given number of layers
        virtual void prepare_for_layers(index_t num_layers) = 0;

        // Accessors
        scalar_t learning_rate() const { return learning_rate_; }
        void set_learning_rate(scalar_t lr) { learning_rate_ = lr; }
        scalar_t weight_decay() const { return weight_decay_; }
    };
}
```

### SGD Optimizer

Simple Stochastic Gradient Descent with optional weight decay.

```cpp
namespace gnnmath::gnn {
    class sgd_optimizer : public optimizer {
    public:
        sgd_optimizer(scalar_t learning_rate, scalar_t weight_decay = 0.0);

        void update(dense_matrix& W, const dense_matrix& grad_W,
                   feature_t& b, const feature_t& grad_b,
                   index_t layer_idx) override;

        void reset() override {}  // Stateless
        void prepare_for_layers(index_t num_layers) override {}
    };
}
```

**Update Rule:**

```
W = W - lr × (grad_W + weight_decay × W)
b = b - lr × grad_b
```

**Implementation:**

```cpp
void sgd_optimizer::update(dense_matrix& W, const dense_matrix& grad_W,
                           feature_t& b, const feature_t& grad_b,
                           index_t layer_idx) {
    // Update weights with optional L2 regularization
    for (index_t i = 0; i < W.rows(); ++i) {
        for (index_t j = 0; j < W.cols(); ++j) {
            scalar_t grad = grad_W(i, j) + weight_decay_ * W(i, j);
            W(i, j) -= learning_rate_ * grad;
        }
    }

    // Update bias (no regularization on bias)
    for (index_t i = 0; i < b.size(); ++i) {
        b[i] -= learning_rate_ * grad_b[i];
    }
}
```

### Adam Optimizer

Adaptive Moment Estimation with bias correction.

```cpp
namespace gnnmath::gnn {
    struct adam_state {
        dense_matrix m_W;   // First moment (mean) for weights
        dense_matrix v_W;   // Second moment (variance) for weights
        feature_t m_b;      // First moment for bias
        feature_t v_b;      // Second moment for bias
        index_t t;          // Timestep (for bias correction)
    };

    class adam_optimizer : public optimizer {
    private:
        scalar_t beta1_;     // First moment decay (default: 0.9)
        scalar_t beta2_;     // Second moment decay (default: 0.999)
        scalar_t epsilon_;   // Numerical stability (default: 1e-8)
        std::vector<adam_state> states_;

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

**Update Rule:**

```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t           (Update first moment)
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²          (Update second moment)

m̂_t = m_t / (1 - β₁^t)                        (Bias correction)
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε)        (Parameter update)
```

**Implementation:**

```cpp
void adam_optimizer::update(dense_matrix& W, const dense_matrix& grad_W,
                            feature_t& b, const feature_t& grad_b,
                            index_t layer_idx) {
    auto& state = states_[layer_idx];
    state.t++;

    // Bias correction factors
    scalar_t bc1 = 1.0 - std::pow(beta1_, state.t);
    scalar_t bc2 = 1.0 - std::pow(beta2_, state.t);

    // Update weights
    for (index_t i = 0; i < W.rows(); ++i) {
        for (index_t j = 0; j < W.cols(); ++j) {
            scalar_t g = grad_W(i, j) + weight_decay_ * W(i, j);

            // Update moments
            state.m_W(i, j) = beta1_ * state.m_W(i, j) + (1 - beta1_) * g;
            state.v_W(i, j) = beta2_ * state.v_W(i, j) + (1 - beta2_) * g * g;

            // Bias-corrected estimates
            scalar_t m_hat = state.m_W(i, j) / bc1;
            scalar_t v_hat = state.v_W(i, j) / bc2;

            // Update parameter
            W(i, j) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }

    // Similar for bias (without weight decay)
    for (index_t i = 0; i < b.size(); ++i) {
        state.m_b[i] = beta1_ * state.m_b[i] + (1 - beta1_) * grad_b[i];
        state.v_b[i] = beta2_ * state.v_b[i] + (1 - beta2_) * grad_b[i] * grad_b[i];

        scalar_t m_hat = state.m_b[i] / bc1;
        scalar_t v_hat = state.v_b[i] / bc2;

        b[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
}
```

### Optimizer Comparison

| Feature | SGD | Adam |
|---------|-----|------|
| State per layer | None | 4 matrices |
| Adaptive LR | No | Yes |
| Momentum | No | First moment |
| Computation | O(params) | O(params) |
| Memory | O(1) | O(params) |
| Good for | Convex, well-tuned | General, robust |

---

## Loss Functions

### Base Class

```cpp
namespace gnnmath::gnn {
    class loss {
    public:
        virtual ~loss() = default;

        // Compute loss value
        virtual scalar_t compute(const feature_matrix_t& predictions,
                                const feature_matrix_t& targets) = 0;

        // Compute gradient w.r.t. predictions
        virtual feature_matrix_t gradient(const feature_matrix_t& predictions,
                                         const feature_matrix_t& targets) = 0;

        // Loss name for logging
        virtual std::string name() const = 0;
    };
}
```

### MSE Loss (Mean Squared Error)

For regression tasks.

```cpp
namespace gnnmath::gnn {
    class mse_loss : public loss {
    public:
        scalar_t compute(const feature_matrix_t& predictions,
                        const feature_matrix_t& targets) override;

        feature_matrix_t gradient(const feature_matrix_t& predictions,
                                 const feature_matrix_t& targets) override;

        std::string name() const override { return "MSE"; }
    };
}
```

**Formula:**

```
L = (1/n) × Σᵢ Σⱼ (pred[i][j] - target[i][j])²
```

**Gradient:**

```
∂L/∂pred[i][j] = (2/n) × (pred[i][j] - target[i][j])
```

**Implementation:**

```cpp
scalar_t mse_loss::compute(const feature_matrix_t& predictions,
                           const feature_matrix_t& targets) {
    scalar_t sum = 0.0;
    index_t count = 0;

    for (index_t i = 0; i < predictions.size(); ++i) {
        for (index_t j = 0; j < predictions[i].size(); ++j) {
            scalar_t diff = predictions[i][j] - targets[i][j];
            sum += diff * diff;
            count++;
        }
    }

    return sum / count;
}

feature_matrix_t mse_loss::gradient(const feature_matrix_t& predictions,
                                    const feature_matrix_t& targets) {
    index_t n = predictions.size();
    index_t count = n * predictions[0].size();
    feature_matrix_t grad(n);

    for (index_t i = 0; i < n; ++i) {
        grad[i].resize(predictions[i].size());
        for (index_t j = 0; j < predictions[i].size(); ++j) {
            grad[i][j] = 2.0 * (predictions[i][j] - targets[i][j]) / count;
        }
    }

    return grad;
}
```

### Cross-Entropy Loss

For classification tasks.

```cpp
namespace gnnmath::gnn {
    class cross_entropy_loss : public loss {
    public:
        scalar_t compute(const feature_matrix_t& predictions,
                        const feature_matrix_t& targets) override;

        feature_matrix_t gradient(const feature_matrix_t& predictions,
                                 const feature_matrix_t& targets) override;

        std::string name() const override { return "CrossEntropy"; }
    };
}
```

**Formula:**

```
L = -(1/n) × Σᵢ Σⱼ target[i][j] × log(pred[i][j])
```

**Note:** Predictions should be probabilities (e.g., after softmax).

**Implementation:**

```cpp
scalar_t cross_entropy_loss::compute(const feature_matrix_t& predictions,
                                     const feature_matrix_t& targets) {
    scalar_t sum = 0.0;
    index_t count = 0;

    for (index_t i = 0; i < predictions.size(); ++i) {
        for (index_t j = 0; j < predictions[i].size(); ++j) {
            // Clamp to avoid log(0)
            scalar_t p = std::max(predictions[i][j], 1e-15);
            sum -= targets[i][j] * std::log(p);
            count++;
        }
    }

    return sum / count;
}
```

---

## Trainer

The trainer orchestrates the training loop, combining pipeline, optimizer, and loss.

### Class Definition

```cpp
namespace gnnmath::gnn {
    class trainer {
    private:
        std::shared_ptr<pipeline> pipeline_;
        std::unique_ptr<optimizer> optimizer_;
        std::unique_ptr<loss> loss_;

    public:
        // Constructor with optimizer type selection
        trainer(std::shared_ptr<pipeline> pipeline,
                optimizer_type opt_type,
                scalar_t learning_rate,
                scalar_t weight_decay = 0.0);

        // Constructor with custom optimizer and loss
        trainer(std::shared_ptr<pipeline> pipeline,
                std::unique_ptr<optimizer> opt,
                std::unique_ptr<loss> loss_fn);

        // Training step
        scalar_t train_step(const feature_matrix_t& input,
                           const sparse_matrix& adjacency,
                           const feature_matrix_t& targets);

        // Compute loss without updating
        scalar_t compute_loss(const feature_matrix_t& input,
                             const sparse_matrix& adjacency,
                             const feature_matrix_t& targets);

        // Accessors
        void set_learning_rate(scalar_t lr);
        scalar_t learning_rate() const;
        void set_weight_decay(scalar_t wd);
        optimizer& get_optimizer();
        loss& get_loss();
    };
}
```

### Training Step

```cpp
scalar_t trainer::train_step(const feature_matrix_t& input,
                             const sparse_matrix& adjacency,
                             const feature_matrix_t& targets) {
    // 1. Forward pass
    feature_matrix_t predictions = pipeline_->process(input, adjacency);

    // 2. Compute loss
    scalar_t loss_value = loss_->compute(predictions, targets);

    // 3. Compute gradients (simplified backprop through last layer)
    feature_matrix_t grad = loss_->gradient(predictions, targets);

    // 4. Update each layer's parameters
    for (index_t i = 0; i < pipeline_->num_layers(); ++i) {
        layer& l = pipeline_->get_layer(i);

        // Compute layer gradients (simplified)
        dense_matrix grad_W = compute_weight_gradient(l, grad);
        feature_t grad_b = compute_bias_gradient(l, grad);

        // Apply optimizer update
        optimizer_->update(l.W, grad_W, l.b, grad_b, i);
    }

    return loss_value;
}
```

### Training Loop Example

```cpp
// Setup
auto pipeline = std::make_shared<gnnmath::gnn::pipeline>();
pipeline->add_layer(std::make_unique<gcn_layer>(7, 32));
pipeline->add_layer(std::make_unique<gcn_layer>(32, 16));
pipeline->add_layer(std::make_unique<gcn_layer>(16, 1));

// Create trainer with Adam optimizer
gnnmath::gnn::trainer trainer(
    pipeline,
    std::make_unique<adam_optimizer>(0.001, 0.0001),
    std::make_unique<mse_loss>()
);

// Training loop
index_t num_epochs = 100;
for (index_t epoch = 0; epoch < num_epochs; ++epoch) {
    scalar_t total_loss = 0.0;

    for (const auto& batch : data_loader) {
        scalar_t loss = trainer.train_step(
            batch.features,
            batch.adjacency,
            batch.targets
        );
        total_loss += loss;
    }

    if (epoch % 10 == 0) {
        std::cout << "Epoch " << epoch
                  << " Loss: " << total_loss / data_loader.size()
                  << "\n";
    }
}

// Save trained model
pipeline->save("trained_model.bin");
```

---

## Complete Training Example

```cpp
#include <gnnmath/gnnmath.hpp>
#include <iostream>

using namespace gnnmath;
using namespace gnnmath::gnn;

int main() {
    // Load training data
    mesh m;
    m.load_obj("training_mesh.obj");

    auto g = graph::from_mesh(m);
    auto adj = g.to_adjacency_matrix();

    // Create target labels (e.g., importance scores)
    feature_matrix_t targets(g.num_vertices);
    for (index_t i = 0; i < g.num_vertices; ++i) {
        targets[i] = {compute_importance(m, i)};
    }

    // Build pipeline
    auto pipeline = std::make_shared<gnnmath::gnn::pipeline>();
    pipeline->add_layer(std::make_unique<gcn_layer>(7, 64, activation_type::RELU));
    pipeline->add_layer(std::make_unique<gcn_layer>(64, 32, activation_type::RELU));
    pipeline->add_layer(std::make_unique<gcn_layer>(32, 16, activation_type::RELU));
    pipeline->add_layer(std::make_unique<gcn_layer>(16, 1, activation_type::SIGMOID));

    // Create trainer
    trainer t(
        pipeline,
        std::make_unique<adam_optimizer>(0.001, 1e-5),
        std::make_unique<mse_loss>()
    );

    // Training
    std::cout << "Training...\n";
    for (index_t epoch = 0; epoch < 100; ++epoch) {
        scalar_t loss = t.train_step(g.node_features, adj, targets);

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ": loss = " << loss << "\n";
        }
    }

    // Evaluate
    scalar_t final_loss = t.compute_loss(g.node_features, adj, targets);
    std::cout << "Final loss: " << final_loss << "\n";

    // Save model
    pipeline->save("mesh_simplifier.bin");

    // Use for inference
    auto scores = pipeline->process(g.node_features, adj);
    std::cout << "Score for vertex 0: " << scores[0][0] << "\n";

    return 0;
}
```

---

## Performance Tips

1. **Batch Processing**: Process multiple small graphs together
2. **Sparse Operations**: Use normalized adjacency for stable training
3. **Learning Rate**: Start with 0.001 for Adam, 0.01 for SGD
4. **Weight Decay**: Use 1e-4 to 1e-5 for regularization
5. **Layer Width**: Wider layers (64-128) often work better than deep narrow ones
6. **Activation**: ReLU is fast; GELU/Mish may improve accuracy
7. **Gradient Clipping**: Consider clipping if training is unstable

---

## Architecture Guidelines

| Mesh Size | Recommended Architecture |
|-----------|-------------------------|
| < 1K vertices | 7 → 32 → 16 → out |
| 1K - 10K | 7 → 64 → 32 → 16 → out |
| 10K - 100K | 7 → 128 → 64 → 32 → out |
| > 100K | 7 → 256 → 128 → 64 → 32 → out |

Use residual connections for deeper networks (> 4 layers).
