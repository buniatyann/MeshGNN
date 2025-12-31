#ifndef GNNMATH_GNN_TRAINING_HPP
#define GNNMATH_GNN_TRAINING_HPP

#include "../math/vector.hpp"
#include "../math/dense_matrix.hpp"
#include "../math/sparse_matrix.hpp"
#include "pipeline.hpp"
#include "optimizers/optimizer.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/adam.hpp"
#include "losses/loss.hpp"
#include "losses/mse.hpp"
#include "losses/cross_entropy.hpp"
#include <vector>
#include <functional>
#include <unordered_map>
#include <string>
#include <memory>
#include <optional>

namespace gnnmath {
namespace gnn {

using feature_vec = gnnmath::vector::vector;

/// @brief Optimizer types (for backward compatibility)
enum class optimizer_type { SGD, ADAM };

/// @brief Trainer for optimizing GNN pipelines.
class trainer {
public:
    /// @brief Constructs a trainer for a pipeline with shared ownership.
    /// @param pipeline_ptr Shared pointer to the GNN pipeline.
    /// @param learning_rate Learning rate for optimization.
    /// @param opt_type Optimizer type (SGD or ADAM).
    /// @param weight_decay L2 regularization coefficient (default: 0).
    /// @throws std::runtime_error If pipeline_ptr is null or learning_rate is non-positive.
    trainer(std::shared_ptr<pipeline> pipeline_ptr, double learning_rate = 0.01,
            optimizer_type opt_type = optimizer_type::SGD, double weight_decay = 0.0);

    /// @brief Constructs a trainer with custom optimizer and loss.
    /// @param pipeline_ptr Shared pointer to the GNN pipeline.
    /// @param opt Custom optimizer.
    /// @param loss Custom loss function.
    /// @throws std::runtime_error If pipeline_ptr is null.
    trainer(std::shared_ptr<pipeline> pipeline_ptr,
            std::unique_ptr<optimizer> opt,
            std::unique_ptr<loss_function> loss = nullptr);

    /// @brief Legacy constructor for backward compatibility (non-owning).
    /// @param pipeline_ptr Raw pointer to the GNN pipeline (caller retains ownership).
    /// @param learning_rate Learning rate for optimization.
    /// @param opt_type Optimizer type (SGD or ADAM).
    /// @param weight_decay L2 regularization coefficient (default: 0).
    /// @throws std::runtime_error If pipeline_ptr is null or learning_rate is non-positive.
    /// @deprecated Use shared_ptr constructor instead.
    trainer(pipeline* pipeline_ptr, double learning_rate = 0.01,
            optimizer_type opt_type = optimizer_type::SGD, double weight_decay = 0.0);

    /// @brief Computes mean squared error loss.
    /// @param predicted Predicted node features.
    /// @param target Target node features.
    /// @return MSE loss value.
    /// @throws std::runtime_error If dimensions mismatch or result is non-finite.
    double mse_loss(const std::vector<feature_vec>& predicted,
                    const std::vector<feature_vec>& target) const;

    /// @brief Computes cross-entropy loss for classification.
    /// @param predicted Predicted probabilities (after softmax).
    /// @param target Target class indices (one-hot or probabilities).
    /// @return Cross-entropy loss value.
    /// @throws std::runtime_error If dimensions mismatch or result is non-finite.
    double cross_entropy_loss(const std::vector<feature_vec>& predicted,
                              const std::vector<feature_vec>& target) const;

    /// @brief Computes loss using the configured loss function.
    /// @param predicted Predicted node features.
    /// @param target Target node features.
    /// @return Loss value.
    double compute_loss(const std::vector<feature_vec>& predicted,
                       const std::vector<feature_vec>& target) const;

    /// @brief Performs one training step with analytic gradients.
    /// @param features Input node features.
    /// @param adj Adjacency matrix.
    /// @param target Target node features.
    /// @throws std::runtime_error If inputs are invalid or pipeline is null.
    void train_step(const std::vector<feature_vec>& features,
                    const matrix::sparse_matrix& adj,
                    const std::vector<feature_vec>& target);

    /// @brief Sets the learning rate.
    /// @param lr New learning rate.
    void set_learning_rate(double lr);

    /// @brief Gets the current learning rate.
    /// @return Current learning rate.
    double learning_rate() const;

    /// @brief Sets the weight decay (L2 regularization).
    /// @param wd New weight decay value.
    void set_weight_decay(double wd);

    /// @brief Gets the optimizer (for advanced configuration).
    /// @return Pointer to the optimizer.
    optimizer* get_optimizer() { return optimizer_.get(); }

    /// @brief Gets the loss function (for advanced configuration).
    /// @return Pointer to the loss function.
    loss_function* get_loss() { return loss_.get(); }

    /// @brief Gets the pipeline (shared ownership).
    /// @return Shared pointer to the pipeline.
    std::shared_ptr<pipeline> get_pipeline() { return pipeline_; }

    /// @brief Gets the pipeline (const access).
    /// @return Const pointer to the pipeline.
    const pipeline* get_pipeline_ptr() const {
        return pipeline_ ? pipeline_.get() : pipeline_raw_;
    }

private:
    /// @brief Computes activation function derivative.
    /// @param x Pre-activation value.
    /// @param act_type Activation type.
    /// @return Derivative value.
    double activation_derivative(double x, activation_type act_type) const;

    /// @brief Internal initialization shared by constructors.
    void init(double learning_rate, optimizer_type opt_type, double weight_decay);

    std::shared_ptr<pipeline> pipeline_;        ///< Shared pointer to the GNN pipeline.
    pipeline* pipeline_raw_ = nullptr;          ///< Raw pointer for legacy compatibility.
    std::unique_ptr<optimizer> optimizer_;      ///< Optimizer instance.
    std::unique_ptr<loss_function> loss_;       ///< Loss function instance.
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_TRAINING_HPP
