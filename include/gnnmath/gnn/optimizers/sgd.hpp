#ifndef GNNMATH_GNN_OPTIMIZERS_SGD_HPP
#define GNNMATH_GNN_OPTIMIZERS_SGD_HPP

#include "optimizer.hpp"

namespace gnnmath {
namespace gnn {

/**
 * @brief Stochastic Gradient Descent optimizer.
 *
 * Updates parameters using: param = param - learning_rate * gradient
 * With optional L2 weight decay: param = param - learning_rate * (gradient + weight_decay * param)
 */
class sgd_optimizer : public optimizer {
public:
    /**
     * @brief Constructs an SGD optimizer.
     * @param learning_rate Initial learning rate.
     * @param weight_decay L2 regularization coefficient (default: 0).
     */
    explicit sgd_optimizer(double learning_rate = 0.01, double weight_decay = 0.0);

    void update(matrix::dense_matrix& weights, feature_vec& bias,
               const matrix::dense_matrix& weight_grad, const feature_vec& bias_grad,
               std::size_t layer_idx) override;

    void reset() override;
    void set_learning_rate(double lr) override { learning_rate_ = lr; }
    double learning_rate() const override { return learning_rate_; }
    void set_weight_decay(double wd) override { weight_decay_ = wd; }
    double weight_decay() const override { return weight_decay_; }
    void prepare_for_layers(std::size_t num_layers) override;

private:
    double learning_rate_;
    double weight_decay_;
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_OPTIMIZERS_SGD_HPP
