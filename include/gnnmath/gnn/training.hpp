#ifndef GNNMATH_GNN_TRAINING_HPP
#define GNNMATH_GNN_TRAINING_HPP

#include "../vector.hpp"
#include "../matrix.hpp"
#include "pipeline.hpp"
#include <vector>
#include <functional>
#include <unordered_map>
#include <string>
#include <memory>
#include <optional>

namespace gnnmath {
namespace gnn {

// Type alias for feature vector (std::vector<double>)
using feature_vec = gnnmath::vector::vector;

/// @brief Optimizer types
enum class optimizer_type { SGD, ADAM };

/// @brief Adam optimizer state for a parameter
struct adam_state {
    std::optional<matrix::dense_matrix> m_weights;  ///< First moment for weights
    std::optional<matrix::dense_matrix> v_weights;  ///< Second moment for weights
    feature_vec m_bias;                             ///< First moment for bias
    feature_vec v_bias;                             ///< Second moment for bias
    std::size_t t = 0;                              ///< Timestep

    adam_state() = default;
};

/// @brief Trainer for optimizing GNN pipelines.
class trainer {
public:
    /// @brief Constructs a trainer for a pipeline.
    /// @param pipeline_ptr Pointer to the GNN pipeline.
    /// @param learning_rate Learning rate for optimization.
    /// @param opt_type Optimizer type (SGD or ADAM).
    /// @param weight_decay L2 regularization coefficient (default: 0).
    /// @throws std::runtime_error If pipeline_ptr is null or learning_rate is non-positive.
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
    void set_learning_rate(double lr) { learning_rate_ = lr; }

    /// @brief Gets the current learning rate.
    /// @return Current learning rate.
    double learning_rate() const { return learning_rate_; }

    /// @brief Sets the weight decay (L2 regularization).
    /// @param wd New weight decay value.
    void set_weight_decay(double wd) { weight_decay_ = wd; }

private:
    /// @brief Computes activation function derivative.
    /// @param x Pre-activation value.
    /// @param act_type Activation type.
    /// @return Derivative value.
    double activation_derivative(double x, activation_type act_type) const;

    /// @brief Applies Adam optimizer update to parameters.
    /// @param weights Weight matrix to update.
    /// @param bias Bias vector to update.
    /// @param weight_grad Weight gradients.
    /// @param bias_grad Bias gradients.
    /// @param state Adam state for this layer.
    void adam_update(matrix::dense_matrix& weights, feature_vec& bias,
                     const matrix::dense_matrix& weight_grad, const feature_vec& bias_grad,
                     adam_state& state);

    pipeline* pipeline_;                    ///< Pointer to the GNN pipeline.
    double learning_rate_;                  ///< Learning rate for optimization.
    double weight_decay_;                   ///< L2 regularization coefficient.
    optimizer_type opt_type_;               ///< Optimizer type.
    std::vector<adam_state> adam_states_;   ///< Adam states per layer.

    // Adam hyperparameters
    static constexpr double beta1_ = 0.9;
    static constexpr double beta2_ = 0.999;
    static constexpr double epsilon_ = 1e-8;
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_TRAINING_HPP
