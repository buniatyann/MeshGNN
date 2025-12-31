#ifndef GNNMATH_GNN_OPTIMIZERS_ADAM_HPP
#define GNNMATH_GNN_OPTIMIZERS_ADAM_HPP

#include "optimizer.hpp"
#include <vector>
#include <optional>

namespace gnnmath {
namespace gnn {

/**
 * @brief Adam optimizer state for a single layer.
 */
struct adam_state {
    std::optional<matrix::dense_matrix> m_weights;  ///< First moment for weights
    std::optional<matrix::dense_matrix> v_weights;  ///< Second moment for weights
    feature_vec m_bias;                             ///< First moment for bias
    feature_vec v_bias;                             ///< Second moment for bias
    std::size_t t = 0;                              ///< Timestep

    adam_state() = default;
};

/**
 * @brief Adam optimizer (Adaptive Moment Estimation).
 *
 * Combines momentum (first moment) and RMSprop (second moment) for adaptive
 * learning rates per parameter.
 */
class adam_optimizer : public optimizer {
public:
    /**
     * @brief Constructs an Adam optimizer.
     * @param learning_rate Initial learning rate.
     * @param weight_decay L2 regularization coefficient (default: 0).
     * @param beta1 Exponential decay rate for first moment (default: 0.9).
     * @param beta2 Exponential decay rate for second moment (default: 0.999).
     * @param epsilon Small constant for numerical stability (default: 1e-8).
     */
    explicit adam_optimizer(double learning_rate = 0.001,
                           double weight_decay = 0.0,
                           double beta1 = 0.9,
                           double beta2 = 0.999,
                           double epsilon = 1e-8);

    void update(matrix::dense_matrix& weights, feature_vec& bias,
               const matrix::dense_matrix& weight_grad, const feature_vec& bias_grad,
               std::size_t layer_idx) override;

    void reset() override;
    void set_learning_rate(double lr) override { learning_rate_ = lr; }
    double learning_rate() const override { return learning_rate_; }
    void set_weight_decay(double wd) override { weight_decay_ = wd; }
    double weight_decay() const override { return weight_decay_; }
    void prepare_for_layers(std::size_t num_layers) override;

    // Adam-specific getters
    double beta1() const { return beta1_; }
    double beta2() const { return beta2_; }
    double epsilon() const { return epsilon_; }

private:
    double learning_rate_;
    double weight_decay_;
    double beta1_;
    double beta2_;
    double epsilon_;
    std::vector<adam_state> states_;
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_OPTIMIZERS_ADAM_HPP
