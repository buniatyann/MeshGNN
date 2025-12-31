#include <gnnmath/gnn/optimizers/sgd.hpp>

namespace gnnmath {
namespace gnn {

sgd_optimizer::sgd_optimizer(double learning_rate, double weight_decay)
    : learning_rate_(learning_rate), weight_decay_(weight_decay) {}

void sgd_optimizer::update(matrix::dense_matrix& weights, feature_vec& bias,
                          const matrix::dense_matrix& weight_grad, const feature_vec& bias_grad,
                          std::size_t /*layer_idx*/) {
    // Update weights: W = W - lr * (grad + weight_decay * W)
    for (std::size_t i = 0; i < weights.rows(); ++i) {
        for (std::size_t j = 0; j < weights.cols(); ++j) {
            double grad = weight_grad(i, j);
            if (weight_decay_ > 0) {
                grad += weight_decay_ * weights(i, j);
            }
         
            weights(i, j) -= learning_rate_ * grad;
        }
    }

    // Update bias: b = b - lr * grad
    for (std::size_t i = 0; i < bias.size(); ++i) {
        bias[i] -= learning_rate_ * bias_grad[i];
    }
}

void sgd_optimizer::reset() {
    // SGD has no state to reset
    return;
}

void sgd_optimizer::prepare_for_layers(std::size_t /*num_layers*/) {
    // SGD has no per-layer state
    return;
}

} // namespace gnn
} // namespace gnnmath
