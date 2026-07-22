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
    if (predicted.empty()) {
        throw std::runtime_error("mse_loss: empty input");
    }

    return gnn::mse_loss{}.compute(predicted, target);
}

double trainer::cross_entropy_loss(const std::vector<feature_vec>& predicted,
                                   const std::vector<feature_vec>& target) const {
    if (predicted.empty()) {
        throw std::runtime_error("cross_entropy_loss: empty input");
    }

    return gnn::cross_entropy_loss{}.compute(predicted, target);
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
    const std::size_t num_layers = layers.size();
    const std::size_t num_nodes = features.size();

    // Forward pass: each layer caches whatever its backward pass needs
    std::vector<std::vector<feature_vec>> activations(num_layers + 1);
    std::vector<std::unique_ptr<layer_cache>> caches(num_layers);
    activations[0] = features;
    for (std::size_t l = 0; l < num_layers; ++l) {
        activations[l + 1] = layers[l]->forward_cached(activations[l], adj, caches[l]);
    }

    // Output gradient from the configured loss function
    std::vector<feature_vec> delta;
    if (loss_) {
        delta = loss_->gradient(activations[num_layers], target);
    }
    else {
        // Fallback: MSE gradient
        delta.resize(num_nodes);
        for (std::size_t i = 0; i < num_nodes; ++i) {
            delta[i] = vec_ops::scalar_multiply(
                vec_ops::operator-(activations[num_layers][i], target[i]),
                2.0 / static_cast<double>(num_nodes));
        }
    }

    // Backpropagate through layers in reverse; parameters are updated only
    // after each layer computed its input gradient with pre-update weights
    for (int l = static_cast<int>(num_layers) - 1; l >= 0; --l) {
        param_refs params = layers[l]->parameters();
        if (!params.weights || !params.bias) {
            continue;
        }

        matrix::dense_matrix weight_grad(layers[l]->in_features(), layers[l]->out_features());
        feature_vec bias_grad(layers[l]->out_features(), 0.0);
        auto delta_prev = layers[l]->backward(delta, activations[l], adj, *caches[l],
                                              weight_grad, bias_grad, l > 0);

        optimizer_->update(*params.weights, *params.bias, weight_grad, bias_grad,
                           static_cast<std::size_t>(l));
        if (l > 0) {
            delta = std::move(delta_prev);
        }
    }
}

} // namespace gnn
} // namespace gnnmath
