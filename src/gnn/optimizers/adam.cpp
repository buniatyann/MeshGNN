#include <gnnmath/gnn/optimizers/adam.hpp>
#include <cmath>

namespace gnnmath {
namespace gnn {

adam_optimizer::adam_optimizer(double learning_rate, double weight_decay,
                               double beta1, double beta2, double epsilon)
    : learning_rate_(learning_rate)
    , weight_decay_(weight_decay)
    , beta1_(beta1)
    , beta2_(beta2)
    , epsilon_(epsilon) {}

void adam_optimizer::update(matrix::dense_matrix& weights, feature_vec& bias,
                           const matrix::dense_matrix& weight_grad, const feature_vec& bias_grad,
                           std::size_t layer_idx) {
    // Ensure we have state for this layer
    if (layer_idx >= states_.size()) {
        states_.resize(layer_idx + 1);
    }

    adam_state& state = states_[layer_idx];
    ++state.t;

    // Initialize moments if needed
    if (!state.m_weights.has_value()) {
        state.m_weights = matrix::dense_matrix(weights.rows(), weights.cols());
        state.v_weights = matrix::dense_matrix(weights.rows(), weights.cols());
        state.m_bias.resize(bias.size(), 0.0);
        state.v_bias.resize(bias.size(), 0.0);
    }

    // Bias correction factors
    double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(state.t));
    double bias_correction2 = 1.0 - std::pow(beta2_, static_cast<double>(state.t));
    for (std::size_t i = 0; i < weights.rows(); ++i) {
        for (std::size_t j = 0; j < weights.cols(); ++j) {
            double grad = weight_grad(i, j);
            if (weight_decay_ > 0) {
                grad += weight_decay_ * weights(i, j);
            }

            // Update biased first moment estimate
            state.m_weights.value()(i, j) = beta1_ * state.m_weights.value()(i, j) + (1.0 - beta1_) * grad;
            // Update biased second raw moment estimate
            state.v_weights.value()(i, j) = beta2_ * state.v_weights.value()(i, j) + (1.0 - beta2_) * grad * grad;

            // Compute bias-corrected estimates
            double m_hat = state.m_weights.value()(i, j) / bias_correction1;
            double v_hat = state.v_weights.value()(i, j) / bias_correction2;

            // Update weights
            weights(i, j) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }

    for (std::size_t i = 0; i < bias.size(); ++i) {
        double grad = bias_grad[i];

        // Update biased first moment estimate
        state.m_bias[i] = beta1_ * state.m_bias[i] + (1.0 - beta1_) * grad;
        // Update biased second raw moment estimate
        state.v_bias[i] = beta2_ * state.v_bias[i] + (1.0 - beta2_) * grad * grad;

        // Compute bias-corrected estimates
        double m_hat = state.m_bias[i] / bias_correction1;
        double v_hat = state.v_bias[i] / bias_correction2;

        // Update bias
        bias[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
}

void adam_optimizer::reset() {
    states_.clear();
}

void adam_optimizer::prepare_for_layers(std::size_t num_layers) {
    if (states_.size() < num_layers) {
        states_.resize(num_layers);
    }
}

} // namespace gnn
} // namespace gnnmath
