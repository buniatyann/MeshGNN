#include <gnnmath/gnn/losses/cross_entropy.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace gnnmath {
namespace gnn {

double cross_entropy_loss::compute(const std::vector<feature_vec>& predicted,
                                   const std::vector<feature_vec>& target) const {
    if (predicted.size() != target.size()) {
        throw std::runtime_error("cross_entropy_loss: predicted and target size mismatch");
    }
    if (predicted.empty()) {
        return 0.0;
    }

    constexpr double epsilon = 1e-15;  // For numerical stability
    double total_loss = 0.0;

    for (std::size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i].size() != target[i].size()) {
            throw std::runtime_error("cross_entropy_loss: feature dimension mismatch at index " + std::to_string(i));
        }
        for (std::size_t j = 0; j < predicted[i].size(); ++j) {
            // Clamp predicted values to avoid log(0)
            double p = std::clamp(predicted[i][j], epsilon, 1.0 - epsilon);
            total_loss -= target[i][j] * std::log(p);
        }
    }

    double result = total_loss / static_cast<double>(predicted.size());
    if (!std::isfinite(result)) {
        throw std::runtime_error("cross_entropy_loss: result is non-finite");
    }

    return result;
}

std::vector<feature_vec> cross_entropy_loss::gradient(const std::vector<feature_vec>& predicted,
                                                      const std::vector<feature_vec>& target) const {
    if (predicted.size() != target.size()) {
        throw std::runtime_error("cross_entropy_loss gradient: predicted and target size mismatch");
    }

    constexpr double epsilon = 1e-15;
    std::vector<feature_vec> grad(predicted.size());
    double scale = 1.0 / static_cast<double>(predicted.size());
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        grad[i].resize(predicted[i].size());
        for (std::size_t j = 0; j < predicted[i].size(); ++j) {
            // Clamp predicted values to avoid division by zero
            double p = std::clamp(predicted[i][j], epsilon, 1.0 - epsilon);
            grad[i][j] = -scale * target[i][j] / p;
        }
    }

    return grad;
}

} // namespace gnn
} // namespace gnnmath
