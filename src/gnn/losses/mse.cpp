#include <gnnmath/gnn/losses/mse.hpp>
#include <cmath>
#include <stdexcept>

namespace gnnmath {
namespace gnn {

double mse_loss::compute(const std::vector<feature_vec>& predicted,
                        const std::vector<feature_vec>& target) const {
    if (predicted.size() != target.size()) {
        throw std::runtime_error("mse_loss: predicted and target size mismatch");
    }
    if (predicted.empty()) {
        return 0.0;
    }

    double total_loss = 0.0;
    std::size_t total_elements = 0;

    for (std::size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i].size() != target[i].size()) {
            throw std::runtime_error("mse_loss: feature dimension mismatch at index " + std::to_string(i));
        }
        for (std::size_t j = 0; j < predicted[i].size(); ++j) {
            double diff = predicted[i][j] - target[i][j];
            total_loss += diff * diff;
            ++total_elements;
        }
    }

    double result = total_loss / static_cast<double>(total_elements);
    if (!std::isfinite(result)) {
        throw std::runtime_error("mse_loss: result is non-finite");
    }

    return result;
}

std::vector<feature_vec> mse_loss::gradient(const std::vector<feature_vec>& predicted,
                                            const std::vector<feature_vec>& target) const {
    if (predicted.size() != target.size()) {
        throw std::runtime_error("mse_loss gradient: predicted and target size mismatch");
    }

    std::size_t total_elements = 0;
    for (const auto& p : predicted) {
        total_elements += p.size();
    }

    std::vector<feature_vec> grad(predicted.size());
    double scale = 2.0 / static_cast<double>(total_elements);
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        grad[i].resize(predicted[i].size());
        for (std::size_t j = 0; j < predicted[i].size(); ++j) {
            grad[i][j] = scale * (predicted[i][j] - target[i][j]);
        }
    }

    return grad;
}

} // namespace gnn
} // namespace gnnmath
