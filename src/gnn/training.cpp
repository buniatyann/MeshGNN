#include <gnnmath/gnn/training.hpp>
#include <gnnmath/math/vector.hpp>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <algorithm>

namespace vec_ops = gnnmath::vector;

namespace gnnmath {
namespace gnn {

using feature_vec = vec_ops::vector;

pipeline* trainer_get_pipeline(const std::shared_ptr<pipeline>& shared, pipeline* raw) {
    return shared ? shared.get() : raw;
}

void trainer::init(double learning_rate, optimizer_type opt_type, double weight_decay) {
    if (learning_rate <= 0.0) {
        throw std::runtime_error("trainer: learning rate must be positive");
    }
    if (weight_decay < 0.0) {
        throw std::runtime_error("trainer: weight decay must be non-negative");
    }

    // optimizer based on type
    if (opt_type == optimizer_type::ADAM) {
        optimizer_ = std::make_unique<adam_optimizer>(learning_rate, weight_decay);
    } 
    else {
        optimizer_ = std::make_unique<sgd_optimizer>(learning_rate, weight_decay);
    }

    pipeline* p = trainer_get_pipeline(pipeline_, pipeline_raw_);
    optimizer_->prepare_for_layers(p->layers().size());

    // Default to MSE loss (use create_loss to avoid name collision with method)
    loss_ = create_loss("mse");
}

trainer::trainer(std::shared_ptr<pipeline> pipeline_ptr, double learning_rate,
                 optimizer_type opt_type, double weight_decay)
    : pipeline_(std::move(pipeline_ptr)), pipeline_raw_(nullptr) {
    if (!pipeline_) {
        throw std::runtime_error("trainer: null pipeline");
    }

    init(learning_rate, opt_type, weight_decay);
}

trainer::trainer(std::shared_ptr<pipeline> pipeline_ptr,
                std::unique_ptr<optimizer> opt,
                std::unique_ptr<loss_function> loss)
    : pipeline_(std::move(pipeline_ptr))
    , pipeline_raw_(nullptr)
    , optimizer_(std::move(opt))
    , loss_(std::move(loss)) {
    if (!pipeline_) {
        throw std::runtime_error("trainer: null pipeline");
    }
    if (!optimizer_) {
        throw std::runtime_error("trainer: null optimizer");
    }

    optimizer_->prepare_for_layers(pipeline_->layers().size());

    // Default to MSE loss if not provided
    if (!loss_) {
        loss_ = create_loss("mse");
    }
}

trainer::trainer(pipeline* pipeline_ptr, double learning_rate,
                 optimizer_type opt_type, double weight_decay)
    : pipeline_(nullptr), pipeline_raw_(pipeline_ptr) {
    if (!pipeline_ptr) {
        throw std::runtime_error("trainer: null pipeline");
    }
    
    init(learning_rate, opt_type, weight_decay);
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

double trainer::compute_loss(const std::vector<feature_vec>& predicted,
                            const std::vector<feature_vec>& target) const {
    if (loss_) {
        return loss_->compute(predicted, target);
    }

    return mse_loss(predicted, target);
}

void trainer::set_learning_rate(double lr) {
    if (optimizer_) {
        optimizer_->set_learning_rate(lr);
    }
}

double trainer::learning_rate() const {
    return optimizer_ ? optimizer_->learning_rate() : 0.0;
}

void trainer::set_weight_decay(double wd) {
    if (optimizer_) {
        optimizer_->set_weight_decay(wd);
    }
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

void trainer::train_step(const std::vector<feature_vec>& features,
                         const matrix::sparse_matrix& adj,
                         const std::vector<feature_vec>& target) {
    pipeline* p = trainer_get_pipeline(pipeline_, pipeline_raw_);
    if (!p) {
        throw std::runtime_error("train_step: null pipeline");
    }
    if (features.empty() || features.size() != adj.rows || target.empty() ||
        target.size() != features.size()) {
        throw std::runtime_error("train_step: invalid input dimensions");
    }

    const auto& layers = p->layers();
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
        } 
        else {
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

    std::size_t layer_idx = num_layers;

    // Backpropagate through layers in reverse
    for (int l = static_cast<int>(num_layers) - 1; l >= 0; --l) {
        auto* gcn = dynamic_cast<gcn_layer*>(layers[l].get());
        auto* edge_conv = dynamic_cast<edge_conv_layer*>(layers[l].get());
        if (!gcn && !edge_conv) {
            continue;
        }

        --layer_idx;
        std::size_t out_dim = gcn ? gcn->out_features() : edge_conv->out_features();
        std::size_t in_dim = gcn ? gcn->in_features() : edge_conv->in_features();
        matrix::dense_matrix& weights = gcn ? gcn->weights() : edge_conv->weights();
        feature_vec& bias = gcn ? gcn->bias() : edge_conv->bias();

        // activation gradients
        activation_type act_type = activation_type::MISH;
        std::vector<feature_vec> delta_pre(features.size(), feature_vec(out_dim, 0.0));
        for (std::size_t i = 0; i < features.size(); ++i) {
            for (std::size_t m = 0; m < out_dim; ++m) {
                double act_deriv = activation_derivative(pre_activations[l][i][m], act_type);
                delta_pre[i][m] = delta[i][m] * act_deriv;
            }
        }

        // weight and bias gradients
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
        } 
        else {
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

        optimizer_->update(weights, bias, weight_grad, bias_grad, layer_idx);

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
            } 
            else {
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
