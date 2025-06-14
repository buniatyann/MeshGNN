#include "../../include/gnnmath/gnn/training.hpp"
#include <cmath>
#include <stdexcept>

namespace gnnmath {
namespace gnn {

trainer::trainer(pipeline* pipeline_ptr, double learning_rate)
    : pipeline_(pipeline_ptr), learning_rate_(learning_rate) {
    if (!pipeline_ptr) {
        throw std::runtime_error("trainer: null pipeline");
    }
    if (learning_rate <= 0.0) {
        throw std::runtime_error("trainer: learning rate must be positive");
    }
}

double trainer::mse_loss(const std::vector<vector>& predicted,
                         const std::vector<vector>& target) const {
    if (predicted.size() != target.size() || predicted.empty() ||
        (!target.empty() && predicted[0].size() != target[0].size())) {
        throw std::runtime_error("mse_loss: dimension mismatch");
    }
    
    double sum = 0.0;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        auto diff = vector::operator-(predicted[i], target[i]);
        sum += vector::dot_product(diff, diff);
        if (!std::isfinite(sum)) {
            throw std::runtime_error("mse_loss: non-finite result");
        }
    }
    
    return sum / static_cast<double>(predicted.size());
}

void trainer::train_step(const std::vector<vector>& features,
                        const matrix::sparse_matrix& adj,
                        const std::vector<vector>& target) {
    if (!pipeline_) {
        throw std::runtime_error("train_step: null pipeline");
    }
    if (features.empty() || features.size() != adj.rows || target.empty() ||
        target.size() != features.size()) {
        throw std::runtime_error("train_step: invalid input dimensions");
    }
    
    const double epsilon = 1e-6;
    const auto& layers = pipeline_->layers();
    
    auto predicted = pipeline_->process(features, adj);
    double base_loss = mse_loss(predicted, target);
    
    for (auto& layer_ptr : layers) {
        auto* gcn = dynamic_cast<gcn_layer*>(layer_ptr.get());
        auto* edge_conv = dynamic_cast<edge_conv_layer*>(layer_ptr.get());
        
        if (gcn || edge_conv) {
            matrix::dense_matrix& weights = gcn ? gcn->weights() : edge_conv->weights();
            vector& bias = gcn ? gcn->bias() : edge_conv->bias();
            
            matrix::dense_matrix weight_grad(weights.rows(), weights.cols());
            for (std::size_t i = 0; i < weights.rows(); ++i) {
                for (std::size_t j = 0; j < weights.cols(); ++j) {
                    double original = weights(i, j);
                    weights(i, j) += epsilon;
                    predicted = pipeline_->process(features, adj);
                    double loss_plus = mse_loss(predicted, target);
                    weights(i, j) = original - epsilon;
                    predicted = pipeline_->process(features, adj);
                    double loss_minus = mse_loss(predicted, target);
                    weights(i, j) = original;
                    weight_grad(i, j) = (loss_plus - loss_minus) / (2.0 * epsilon);
                    if (!std::isfinite(weight_grad(i, j))) {
                        throw std::runtime_error("train_step: non-finite gradient");
                    }
                }
            }
            
            for (std::size_t i = 0; i < weights.rows(); ++i) {
                for (std::size_t j = 0; j < weights.cols(); ++j) {
                    weights(i, j) -= learning_rate_ * weight_grad(i, j);
                }
            }
            
            vector bias_grad(bias.size(), 0.0);
            for (std::size_t i = 0; i < bias.size(); ++i) {
                double original = bias[i];
                bias[i] += epsilon;
                predicted = pipeline_->process(features, adj);
                double loss_plus = mse_loss(predicted, target);
                bias[i] = original - epsilon;
                predicted = pipeline_->process(features, adj);
                double loss_minus = mse_loss(predicted, target);
                bias[i] = original;
                bias_grad[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
                if (!std::isfinite(bias_grad[i])) {
                    throw std::runtime_error("train_step: non-finite gradient");
                }
            }
            
            for (std::size_t i = 0; i < bias.size(); ++i) {
                bias[i] -= learning_rate_ * bias_grad[i];
            }
        }
    }
}
}