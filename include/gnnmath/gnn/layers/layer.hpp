#ifndef GNNMATH_GNN_LAYERS_LAYER_HPP
#define GNNMATH_GNN_LAYERS_LAYER_HPP

#include "../../math/dense_matrix.hpp"
#include "../../math/sparse_matrix.hpp"
#include "../../math/vector.hpp"
#include "../../core/types.hpp"
#include <vector>
#include <memory>
#include <cstdint>

namespace gnnmath {

using feature_vec = gnnmath::vector::vector;

namespace gnn {

/// @brief Activation function types.
enum class activation_type { RELU, MISH, SIGMOID, GELU };

/// @brief Layer type tags (also used as the model file type byte).
enum class layer_kind : std::uint8_t { unknown = 0, gcn = 1, edge_conv = 2 };

/// @brief Opaque per-layer forward cache; concrete layers define derived structs
/// holding whatever intermediates their backward pass needs.
struct layer_cache {
    virtual ~layer_cache() = default;
};

/// @brief Non-owning view of a layer's trainable parameters (null for
/// parameter-less layers).
struct param_refs {
    matrix::dense_matrix* weights = nullptr;
    feature_vec* bias = nullptr;
};

/// @brief Applies an activation function element-wise.
/// @param v Input vector.
/// @param act Activation type.
/// @return Activated vector.
feature_vec apply_activation(const feature_vec& v, activation_type act);

/// @brief Computes the activation function derivative.
/// @param x Pre-activation value.
/// @param act Activation type.
/// @return Derivative value at x.
double activation_derivative(double x, activation_type act);

/// @brief Abstract base class for GNN layers.
class layer {
public:
    /// @brief Virtual destructor for polymorphism.
    virtual ~layer() = default;

    /// @brief Performs forward pass through the layer.
    /// @param features Input node features (rows: nodes, cols: feature dim).
    /// @param adj Adjacency matrix in CSR format.
    /// @return Output node features.
    /// @throws std::runtime_error If dimensions are incompatible.
    virtual std::vector<feature_vec> forward(const std::vector<feature_vec>& features,
                                       const matrix::sparse_matrix& adj) const = 0;

    /// @brief Forward pass that also fills a cache for a later backward pass.
    /// Default implementation: plain forward, null cache (for layers without
    /// trainable parameters).
    /// @param features Input node features.
    /// @param adj Adjacency matrix.
    /// @param cache Receives the layer's forward cache.
    /// @return Output node features.
    virtual std::vector<feature_vec> forward_cached(const std::vector<feature_vec>& features,
                                              const matrix::sparse_matrix& adj,
                                              std::unique_ptr<layer_cache>& cache) const;

    /// @brief Backward pass: consumes the loss gradient w.r.t. this layer's
    /// output and produces parameter gradients plus the gradient w.r.t. the
    /// layer's input.
    /// @param delta_out Gradient w.r.t. this layer's output (per node).
    /// @param input The features this layer was forwarded with.
    /// @param adj Adjacency matrix used in the forward pass.
    /// @param cache Cache produced by forward_cached.
    /// @param weight_grad Receives the weight gradient (in_features x out_features).
    /// @param bias_grad Receives the bias gradient (out_features).
    /// @param compute_delta_prev Skip computing the input gradient when false
    /// (e.g. for the first layer).
    /// @return Gradient w.r.t. the input (empty when compute_delta_prev is false).
    /// @throws std::runtime_error If the layer does not support training.
    virtual std::vector<feature_vec> backward(const std::vector<feature_vec>& delta_out,
                                        const std::vector<feature_vec>& input,
                                        const matrix::sparse_matrix& adj,
                                        const layer_cache& cache,
                                        matrix::dense_matrix& weight_grad,
                                        feature_vec& bias_grad,
                                        bool compute_delta_prev) const;

    /// @brief Returns views of the trainable parameters (nulls if none).
    virtual param_refs parameters() { return {}; }

    /// @brief Returns the layer type tag.
    virtual layer_kind kind() const { return layer_kind::unknown; }

    /// @brief Returns the layer's activation type (RELU for layers without one).
    virtual activation_type act() const { return activation_type::RELU; }

    /// @brief Returns the input feature dimension.
    /// @return Input dimension.
    virtual std::size_t in_features() const = 0;

    /// @brief Returns the output feature dimension.
    /// @return Output dimension.
    virtual std::size_t out_features() const = 0;
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_LAYERS_LAYER_HPP
