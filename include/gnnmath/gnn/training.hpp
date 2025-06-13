#ifndef GNNMATH_GNN_TRAINING_HPP
#define GNNMATH_GNN_TRAINING_HPP

#include "../vector.hpp" 
#include "../matrix.hpp"
#include "pipeline.hpp"
#include <vector>

namespace gnnmath {
namespace gnn {

/// @brief Trainer for optimizing GNN pipelines.
class trainer {
public:
    /// @brief Constructs a trainer for a pipeline.
    /// @param pipeline_ptr Pointer to the GNN pipeline.
    /// @param learning_rate Learning rate for SGD.
    /// @throws std::runtime_error If pipeline_ptr is null or learning_rate is non-positive.
    trainer(pipeline* pipeline_ptr, double learning_rate = 0.01);

    /// @brief Computes mean squared error loss.
    /// @param predicted Predicted node features.
    /// @param target Target node features.
    /// @return MSE loss value.
    /// @throws std::runtime_error If dimensions mismatch or result is non-finite.
    double mse_loss(const std::vector<vector>& predicted,
                    const std::vector<vector>& target) const;

    /// @brief Performs one training step using SGD with numerical gradients.
    /// @param features Input node features.
    /// @param adj Adjacency matrix.
    /// @param target Target node features.
    /// @throws std::runtime_error If inputs are invalid or pipeline is null.
    void train_step(const std::vector<vector>& features,
                    const matrix::sparse_matrix& adj,
                    const std::vector<vector>& target);

private:
    pipeline* pipeline_;         ///< Pointer to the GNN pipeline.
    double learning_rate_;       ///< Learning rate for SGD.
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_TRAINING_HPP