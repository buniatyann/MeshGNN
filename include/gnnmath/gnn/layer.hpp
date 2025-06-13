#ifndef GNNMATH_GNN_LAYER_HPP
#define GNNMATH_GNN_LAYER_HPP

#include "../matrix.hpp"
#include "../vector.hpp"
#include "../types.hpp"
#include <vector>

namespace gnnmath {
using vector = gnnmath::vector;

namespace gnn {

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
    virtual std::vector<vector> forward(const std::vector<vector>& features,
                                       const matrix::sparse_matrix& adj) const = 0;

    /// @brief Returns the input feature dimension.
    /// @return Input dimension.
    virtual std::size_t in_features() const = 0;

    /// @brief Returns the output feature dimension.
    /// @return Output dimension.
    virtual std::size_t out_features() const = 0;
};

/// @brief Activation function types.
enum class activation_type { RELU, MISH, SIGMOID, GELU };

/// @brief Graph Convolutional Network (GCN) layer for message passing.
class gcn_layer : public layer {
public:
    /// @brief Constructs a GCN layer with random weights.
    /// @param in_dim Input feature dimension.
    /// @param out_dim Output feature dimension.
    /// @param activation Activation function type (default: MISH).
    /// @throws std::runtime_error If dimensions are zero.
    gcn_layer(std::size_t in_dim, std::size_t out_dim, activation_type activation = activation_type::MISH);

    /// @brief Forward pass: H' = activation(A * H * W + b).
    /// @param features Input node features.
    /// @param adj Adjacency matrix.
    /// @return Output node features.
    /// @throws std::runtime_error If dimensions mismatch or result is non-finite.
    std::vector<vector> forward(const std::vector<vector>& features,
                               const matrix::sparse_matrix& adj) const override;

    /// @brief Returns input feature dimension.
    /// @return Input dimension.
    std::size_t in_features() const override { return in_dim_; }

    /// @brief Returns output feature dimension.
    /// @return Output dimension.
    std::size_t out_features() const override { return out_dim_; }

    /// @brief Accesses weights for training.
    /// @return Reference to weight matrix.
    matrix::dense_matrix& weights() { return weights_; }

    /// @brief Accesses bias for training.
    /// @return Reference to bias vector.
    vector& bias() { return bias_; }

private:
    std::size_t in_dim_;           ///< Input feature dimension.
    std::size_t out_dim_;          ///< Output feature dimension.
    matrix::dense_matrix weights_; ///< Weight matrix (in_dim x out_dim).
    vector bias_;                  ///< Bias vector (out_dim).
    activation_type activation_;   ///< Activation function type.
};

/// @brief Edge-conditioned convolution layer for edge feature aggregation.
class edge_conv_layer : public layer {
public:
    /// @brief Constructs an edge-conditioned layer with random weights.
    /// @param in_dim Input feature dimension.
    /// @param out_dim Output feature dimension.
    /// @param activation Activation function type (default: MISH).
    /// @throws std::runtime_error If dimensions are zero.
    edge_conv_layer(std::size_t in_dim, std::size_t out_dim, activation_type activation = activation_type::MISH);

    /// @brief Forward pass: aggregates activation((f_i - f_j) * W + b).
    /// @param features Input node features.
    /// @param adj Adjacency matrix.
    /// @return Output node features.
    /// @throws std::runtime_error If dimensions mismatch or result is non-finite.
    std::vector<vector> forward(const std::vector<vector>& features,
                               const matrix::sparse_matrix& adj) const override;

    /// @brief Returns input feature dimension.
    /// @return Input dimension.
    std::size_t in_features() const override { return in_dim_; }

    /// @brief Returns output feature dimension.
    /// @return Output dimension.
    std::size_t out_features() const override { return out_dim_; }

    /// @brief Accesses weights for training.
    /// @return Reference to weight matrix.
    matrix::dense_matrix& weights() { return weights_; }

    /// @brief Accesses bias for training.
    /// @return Reference to bias vector.
    vector& bias() { return bias_; }

private:
    std::size_t in_dim_;           ///< Input feature dimension.
    std::size_t out_dim_;          ///< Output feature dimension.
    matrix::dense_matrix weights_; ///< Weight matrix (in_dim x out_dim).
    vector bias_;                  ///< Bias vector (out_dim).
    activation_type activation_;   ///< Activation function type.
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_LAYER_HPP