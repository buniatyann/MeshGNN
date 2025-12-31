#ifndef GNNMATH_GNN_LAYERS_LAYER_HPP
#define GNNMATH_GNN_LAYERS_LAYER_HPP

#include "../../math/dense_matrix.hpp"
#include "../../math/sparse_matrix.hpp"
#include "../../math/vector.hpp"
#include "../../core/types.hpp"
#include <vector>

namespace gnnmath {

using feature_vec = gnnmath::vector::vector;

namespace gnn {

/// @brief Activation function types.
enum class activation_type { RELU, MISH, SIGMOID, GELU };

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
