#ifndef GNNMATH_GNN_OPTIMIZERS_OPTIMIZER_HPP
#define GNNMATH_GNN_OPTIMIZERS_OPTIMIZER_HPP

#include "../../math/dense_matrix.hpp"
#include "../../math/vector.hpp"
#include <memory>

namespace gnnmath {
namespace gnn {

using feature_vec = gnnmath::vector::vector;

/**
 * @brief Abstract base class for optimizers.
 *
 * Optimizers update model parameters (weights and biases) based on computed gradients.
 */
class optimizer {
public:
    virtual ~optimizer() = default;

    /**
     * @brief Updates weights and bias using computed gradients.
     * @param weights Weight matrix to update.
     * @param bias Bias vector to update.
     * @param weight_grad Gradient of weights.
     * @param bias_grad Gradient of bias.
     * @param layer_idx Index of the layer (for per-layer state tracking).
     */
    virtual void update(matrix::dense_matrix& weights, feature_vec& bias,
                       const matrix::dense_matrix& weight_grad, const feature_vec& bias_grad,
                       std::size_t layer_idx) = 0;

    /**
     * @brief Resets optimizer state (for new training runs).
     */
    virtual void reset() = 0;

    /**
     * @brief Sets the learning rate.
     * @param lr New learning rate.
     */
    virtual void set_learning_rate(double lr) = 0;

    /**
     * @brief Gets the current learning rate.
     * @return Current learning rate.
     */
    virtual double learning_rate() const = 0;

    /**
     * @brief Sets the weight decay (L2 regularization).
     * @param wd Weight decay coefficient.
     */
    virtual void set_weight_decay(double wd) = 0;

    /**
     * @brief Gets the current weight decay.
     * @return Current weight decay.
     */
    virtual double weight_decay() const = 0;

    /**
     * @brief Ensures optimizer has state for the given number of layers.
     * @param num_layers Number of layers to prepare state for.
     */
    virtual void prepare_for_layers(std::size_t num_layers) = 0;
};

/**
 * @brief Factory function to create optimizers by type name.
 * @param type Optimizer type ("sgd" or "adam").
 * @param learning_rate Initial learning rate.
 * @param weight_decay L2 regularization coefficient.
 * @return Unique pointer to the created optimizer.
 * @throws std::invalid_argument If type is unknown.
 */
std::unique_ptr<optimizer> create_optimizer(const std::string& type,
                                            double learning_rate = 0.01,
                                            double weight_decay = 0.0);

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_OPTIMIZERS_OPTIMIZER_HPP
