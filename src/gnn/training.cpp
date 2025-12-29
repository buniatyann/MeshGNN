#include "../../include/gnnmath/gnn/training.hpp"
#include "../../include/gnnmath/vector.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// Namespace alias for vector operations (must be outside gnnmath namespace)
namespace vec_ops = gnnmath::vector;

namespace gnnmath {
namespace gnn {

// Use the feature_vec type alias
using feature_vec = vec_ops::vector;

trainer::trainer(pipeline* pipeline_ptr, double learning_rate,
                 optimizer_type opt_type, double weight_decay)
    : pipeline_(pipeline_ptr), learning_rate_(learning_rate),
      weight_decay_(weight_decay), opt_type_(opt_type) {
    if (!pipeline_ptr) {
        throw std::runtime_error("trainer: null pipeline");
    }
    if (learning_rate <= 0.0) {
        throw std::runtime_error("trainer: learning rate must be positive");
    }
    if (weight_decay < 0.0) {
        throw std::runtime_error("trainer: weight decay must be non-negative");
    }

    // Initialize Adam states for each layer
    if (opt_type_ == optimizer_type::ADAM) {
        const auto& layers = pipeline_->layers();
        adam_states_.reserve(layers.size());

        for (const auto& layer_ptr : layers) {
            auto* gcn = dynamic_cast<gcn_layer*>(layer_ptr.get());
            auto* edge_conv = dynamic_cast<edge_conv_layer*>(layer_ptr.get());

            if (gcn || edge_conv) {
                std::size_t in_dim = gcn ? gcn->in_features() : edge_conv->in_features();
                std::size_t out_dim = gcn ? gcn->out_features() : edge_conv->out_features();

                adam_state state;
                state.m_weights.emplace(in_dim, out_dim);
                state.v_weights.emplace(in_dim, out_dim);
                state.m_bias = feature_vec(out_dim, 0.0);
                state.v_bias = feature_vec(out_dim, 0.0);
                state.t = 0;
                adam_states_.push_back(std::move(state));
            }
        }
    }
}

double trainer::mse_loss(const std::vector<feature_vec>& predicted,
                         const std::vector<feature_vec>& target) const {
    if (predicted.size() != target.size() || predicted.empty() ||
        (!target.empty() && predicted[0].size() != target[0].size())) {
        throw std::runtime_error("mse_loss: dimension mismatch");
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        auto diff = vec_ops::operator-(predicted[i], target[i]);
        sum += vec_ops::dot_product(diff, diff);
        if (!std::isfinite(sum)) {
            throw std::runtime_error("mse_loss: non-finite result");
        }
    }

    return sum / static_cast<double>(predicted.size());
}

double trainer::cross_entropy_loss(const std::vector<feature_vec>& predicted,
                                   const std::vector<feature_vec>& target) const {
    if (predicted.size() != target.size() || predicted.empty() ||
        (!target.empty() && predicted[0].size() != target[0].size())) {
        throw std::runtime_error("cross_entropy_loss: dimension mismatch");
    }

    constexpr double epsilon = 1e-10;
    double sum = 0.0;

    for (std::size_t i = 0; i < predicted.size(); ++i) {
        for (std::size_t j = 0; j < predicted[i].size(); ++j) {
            // Clamp predicted values to avoid log(0)
            double p = std::max(epsilon, std::min(1.0 - epsilon, predicted[i][j]));
            sum -= target[i][j] * std::log(p);
        }
        if (!std::isfinite(sum)) {
            throw std::runtime_error("cross_entropy_loss: non-finite result");
        }
    }

    return sum / static_cast<double>(predicted.size());
}

double trainer::activation_derivative(double x, activation_type act_type) const {
    switch (act_type) {
        case activation_type::RELU:
            return x > 0.0 ? 1.0 : 0.0;

        case activation_type::SIGMOID: {
            double s = 1.0 / (1.0 + std::exp(-std::min(x, 700.0)));
            return s * (1.0 - s);
        }

        case activation_type::MISH: {
            // Mish: x * tanh(softplus(x))
            // Derivative: tanh(sp) + x * sech^2(sp) * sigmoid(x)
            double sp = std::log1p(std::exp(std::min(x, 700.0)));
            double tanh_sp = std::tanh(sp);
            double sig = 1.0 / (1.0 + std::exp(-std::min(x, 700.0)));
            double sech2 = 1.0 - tanh_sp * tanh_sp;
            return tanh_sp + x * sech2 * sig;
        }

        case activation_type::GELU: {
            // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
            // Derivative: 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
            constexpr double sqrt_2 = 1.4142135623730951;
            constexpr double sqrt_2pi = 2.5066282746310002;
            double erf_term = 0.5 * (1.0 + std::erf(x / sqrt_2));
            double exp_term = std::exp(-0.5 * x * x) / sqrt_2pi;
            return erf_term + x * exp_term;
        }

        default:
            return 1.0;
    }
}

void trainer::adam_update(matrix::dense_matrix& weights, feature_vec& bias,
                          const matrix::dense_matrix& weight_grad, const feature_vec& bias_grad,
                          adam_state& state) {
    state.t++;
    double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(state.t));
    double bias_correction2 = 1.0 - std::pow(beta2_, static_cast<double>(state.t));

    // Update weight moments and apply
    for (std::size_t i = 0; i < weights.rows(); ++i) {
        for (std::size_t j = 0; j < weights.cols(); ++j) {
            double g = weight_grad(i, j);
            // Apply weight decay
            g += weight_decay_ * weights(i, j);

            // Update moments
            (*state.m_weights)(i, j) = beta1_ * (*state.m_weights)(i, j) + (1.0 - beta1_) * g;
            (*state.v_weights)(i, j) = beta2_ * (*state.v_weights)(i, j) + (1.0 - beta2_) * g * g;

            // Bias-corrected estimates
            double m_hat = (*state.m_weights)(i, j) / bias_correction1;
            double v_hat = (*state.v_weights)(i, j) / bias_correction2;

            // Update weights
            weights(i, j) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }

    // Update bias moments and apply
    for (std::size_t i = 0; i < bias.size(); ++i) {
        double g = bias_grad[i];

        // Update moments
        state.m_bias[i] = beta1_ * state.m_bias[i] + (1.0 - beta1_) * g;
        state.v_bias[i] = beta2_ * state.v_bias[i] + (1.0 - beta2_) * g * g;

        // Bias-corrected estimates
        double m_hat = state.m_bias[i] / bias_correction1;
        double v_hat = state.v_bias[i] / bias_correction2;

        // Update bias
        bias[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
}

void trainer::train_step(const std::vector<feature_vec>& features,
                         const matrix::sparse_matrix& adj,
                         const std::vector<feature_vec>& target) {
    if (!pipeline_) {
        throw std::runtime_error("train_step: null pipeline");
    }
    if (features.empty() || features.size() != adj.rows || target.empty() ||
        target.size() != features.size()) {
        throw std::runtime_error("train_step: invalid input dimensions");
    }

    const auto& layers = pipeline_->layers();
    std::size_t num_layers = layers.size();

    // Forward pass: store activations for each layer
    std::vector<std::vector<feature_vec>> activations(num_layers + 1);
    std::vector<std::vector<feature_vec>> pre_activations(num_layers);
    activations[0] = features;

    for (std::size_t l = 0; l < num_layers; ++l) {
        auto* gcn = dynamic_cast<gcn_layer*>(layers[l].get());
        auto* edge_conv = dynamic_cast<edge_conv_layer*>(layers[l].get());

        if (!gcn && !edge_conv) {
            activations[l + 1] = activations[l];
            continue;
        }

        std::size_t out_dim = gcn ? gcn->out_features() : edge_conv->out_features();
        std::size_t in_dim = gcn ? gcn->in_features() : edge_conv->in_features();
        const matrix::dense_matrix& weights = gcn ? gcn->weights() : edge_conv->weights();
        const feature_vec& bias = gcn ? gcn->bias() : edge_conv->bias();

        std::vector<feature_vec> pre_act(features.size(), feature_vec(out_dim, 0.0));
        std::vector<feature_vec> post_act(features.size(), feature_vec(out_dim, 0.0));

        if (gcn) {
            // GCN: H' = A * H * W + b
            for (std::size_t i = 0; i < features.size(); ++i) {
                // Aggregate neighbor features using adjacency
                feature_vec aggregated(in_dim, 0.0);
                for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
                    std::size_t j = adj.col_ind[k];
                    aggregated = vec_ops::operator+(aggregated, activations[l][j]);
                }

                // Linear transform
                for (std::size_t m = 0; m < out_dim; ++m) {
                    double sum = 0.0;
                    for (std::size_t n = 0; n < in_dim; ++n) {
                        sum += aggregated[n] * weights(n, m);
                    }
                    pre_act[i][m] = sum + bias[m];
                }
            }
        } else {
            // EdgeConv: aggregates activation((f_i - f_j) * W + b)
            for (std::size_t i = 0; i < features.size(); ++i) {
                for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
                    std::size_t j = adj.col_ind[k];
                    feature_vec diff = vec_ops::operator-(activations[l][i], activations[l][j]);
                    for (std::size_t m = 0; m < out_dim; ++m) {
                        double sum = 0.0;
                        for (std::size_t n = 0; n < in_dim; ++n) {
                            sum += diff[n] * weights(n, m);
                        }
                        pre_act[i][m] += sum;
                    }
                }
                for (std::size_t m = 0; m < out_dim; ++m) {
                    pre_act[i][m] += bias[m];
                }
            }
        }

        pre_activations[l] = pre_act;

        // Apply activation (default MISH as used in layers)
        activation_type act_type = activation_type::MISH;
        for (std::size_t i = 0; i < features.size(); ++i) {
            switch (act_type) {
                case activation_type::RELU:
                    post_act[i] = vec_ops::relu(pre_act[i]);
                    break;
                case activation_type::MISH:
                    post_act[i] = vec_ops::mish(pre_act[i]);
                    break;
                case activation_type::SIGMOID:
                    post_act[i] = vec_ops::sigmoid(pre_act[i]);
                    break;
                case activation_type::GELU:
                    post_act[i] = vec_ops::gelu(pre_act[i]);
                    break;
            }
        }

        activations[l + 1] = post_act;
    }

    // Backward pass: compute gradients
    // Output gradient: d_loss/d_output = 2 * (output - target) / N for MSE
    std::vector<feature_vec> delta(features.size());
    for (std::size_t i = 0; i < features.size(); ++i) {
        delta[i] = vec_ops::scalar_multiply(
            vec_ops::operator-(activations[num_layers][i], target[i]),
            2.0 / static_cast<double>(features.size()));
    }

    std::size_t adam_idx = adam_states_.size();

    // Backpropagate through layers in reverse
    for (int l = static_cast<int>(num_layers) - 1; l >= 0; --l) {
        auto* gcn = dynamic_cast<gcn_layer*>(layers[l].get());
        auto* edge_conv = dynamic_cast<edge_conv_layer*>(layers[l].get());

        if (!gcn && !edge_conv) {
            continue;
        }

        if (opt_type_ == optimizer_type::ADAM && adam_idx > 0) {
            --adam_idx;
        }

        std::size_t out_dim = gcn ? gcn->out_features() : edge_conv->out_features();
        std::size_t in_dim = gcn ? gcn->in_features() : edge_conv->in_features();
        matrix::dense_matrix& weights = gcn ? gcn->weights() : edge_conv->weights();
        feature_vec& bias = gcn ? gcn->bias() : edge_conv->bias();

        // Compute activation gradients
        activation_type act_type = activation_type::MISH;
        std::vector<feature_vec> delta_pre(features.size(), feature_vec(out_dim, 0.0));
        for (std::size_t i = 0; i < features.size(); ++i) {
            for (std::size_t m = 0; m < out_dim; ++m) {
                double act_deriv = activation_derivative(pre_activations[l][i][m], act_type);
                delta_pre[i][m] = delta[i][m] * act_deriv;
            }
        }

        // Compute weight and bias gradients
        matrix::dense_matrix weight_grad(in_dim, out_dim);
        feature_vec bias_grad(out_dim, 0.0);

        if (gcn) {
            // Weight gradient for GCN
            for (std::size_t i = 0; i < features.size(); ++i) {
                // Aggregate neighbor features
                feature_vec aggregated(in_dim, 0.0);
                for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
                    std::size_t j = adj.col_ind[k];
                    aggregated = vec_ops::operator+(aggregated, activations[l][j]);
                }

                // Accumulate gradients
                for (std::size_t n = 0; n < in_dim; ++n) {
                    for (std::size_t m = 0; m < out_dim; ++m) {
                        weight_grad(n, m) += aggregated[n] * delta_pre[i][m];
                    }
                }

                // Bias gradient
                for (std::size_t m = 0; m < out_dim; ++m) {
                    bias_grad[m] += delta_pre[i][m];
                }
            }
        } else {
            // Weight gradient for EdgeConv
            for (std::size_t i = 0; i < features.size(); ++i) {
                for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
                    std::size_t j = adj.col_ind[k];
                    feature_vec diff = vec_ops::operator-(activations[l][i], activations[l][j]);
                    for (std::size_t n = 0; n < in_dim; ++n) {
                        for (std::size_t m = 0; m < out_dim; ++m) {
                            weight_grad(n, m) += diff[n] * delta_pre[i][m];
                        }
                    }
                }

                // Bias gradient
                for (std::size_t m = 0; m < out_dim; ++m) {
                    bias_grad[m] += delta_pre[i][m];
                }
            }
        }

        // Apply optimizer update
        if (opt_type_ == optimizer_type::ADAM) {
            adam_update(weights, bias, weight_grad, bias_grad, adam_states_[adam_idx]);
        } else {
            // SGD with weight decay
            for (std::size_t i = 0; i < weights.rows(); ++i) {
                for (std::size_t j = 0; j < weights.cols(); ++j) {
                    double grad = weight_grad(i, j) + weight_decay_ * weights(i, j);
                    weights(i, j) -= learning_rate_ * grad;
                }
            }
            for (std::size_t i = 0; i < bias.size(); ++i) {
                bias[i] -= learning_rate_ * bias_grad[i];
            }
        }

        // Compute delta for previous layer (if not first layer)
        if (l > 0) {
            std::vector<feature_vec> delta_prev(features.size(), feature_vec(in_dim, 0.0));
            if (gcn) {
                // Backprop through GCN
                for (std::size_t i = 0; i < features.size(); ++i) {
                    for (std::size_t n = 0; n < in_dim; ++n) {
                        double sum = 0.0;
                        for (std::size_t m = 0; m < out_dim; ++m) {
                            sum += delta_pre[i][m] * weights(n, m);
                        }
                        // Aggregate to neighbors
                        for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
                            std::size_t j = adj.col_ind[k];
                            delta_prev[j][n] += sum;
                        }
                    }
                }
            } else {
                // Backprop through EdgeConv
                for (std::size_t i = 0; i < features.size(); ++i) {
                    for (std::size_t n = 0; n < in_dim; ++n) {
                        double sum = 0.0;
                        for (std::size_t m = 0; m < out_dim; ++m) {
                            sum += delta_pre[i][m] * weights(n, m);
                        }
                        delta_prev[i][n] += sum * static_cast<double>(adj.row_ptr[i + 1] - adj.row_ptr[i]);
                    }
                }
            }
            delta = delta_prev;
        }
    }
}

} // namespace gnn
} // namespace gnnmath
