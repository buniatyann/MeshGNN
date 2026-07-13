#ifndef GNNMATH_GNN_LAYERS_EDGE_CONV_LAYER_HPP
#define GNNMATH_GNN_LAYERS_EDGE_CONV_LAYER_HPP

#include "layer.hpp"

namespace gnnmath {
namespace gnn {

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
    std::vector<feature_vec> forward(const std::vector<feature_vec>& features,
                               const matrix::sparse_matrix& adj) const override;

    /// @brief Forward pass caching per-edge pre-activations for backward.
    std::vector<feature_vec> forward_cached(const std::vector<feature_vec>& features,
                                      const matrix::sparse_matrix& adj,
                                      std::unique_ptr<layer_cache>& cache) const override;

    /// @brief Backward pass through the EdgeConv layer.
    std::vector<feature_vec> backward(const std::vector<feature_vec>& delta_out,
                                const std::vector<feature_vec>& input,
                                const matrix::sparse_matrix& adj,
                                const layer_cache& cache,
                                matrix::dense_matrix& weight_grad,
                                feature_vec& bias_grad,
                                bool compute_delta_prev) const override;

    /// @brief Returns input feature dimension.
    /// @return Input dimension.
    std::size_t in_features() const override { return in_dim_; }

    /// @brief Returns output feature dimension.
    /// @return Output dimension.
    std::size_t out_features() const override { return out_dim_; }

    /// @brief Returns views of weights and bias.
    param_refs parameters() override { return {&weights_, &bias_}; }

    /// @brief Returns the layer type tag.
    layer_kind kind() const override { return layer_kind::edge_conv; }

    /// @brief Returns the activation type.
    activation_type act() const override { return activation_; }

    /// @brief Accesses weights for training.
    /// @return Reference to weight matrix.
    matrix::dense_matrix& weights() { return weights_; }

    /// @brief Accesses weights (const).
    const matrix::dense_matrix& weights() const { return weights_; }

    /// @brief Accesses bias for training.
    /// @return Reference to bias vector.
    feature_vec& bias() { return bias_; }

    /// @brief Accesses bias (const).
    const feature_vec& bias() const { return bias_; }

    /// @brief Returns the activation function type.
    /// @return Activation type.
    activation_type activation() const { return activation_; }

private:
    std::size_t in_dim_;           ///< Input feature dimension.
    std::size_t out_dim_;          ///< Output feature dimension.
    matrix::dense_matrix weights_; ///< Weight matrix (in_dim x out_dim).
    feature_vec bias_;             ///< Bias vector (out_dim).
    activation_type activation_;   ///< Activation function type.
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_LAYERS_EDGE_CONV_LAYER_HPP
