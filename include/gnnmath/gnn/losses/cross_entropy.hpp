#ifndef GNNMATH_GNN_LOSSES_CROSS_ENTROPY_HPP
#define GNNMATH_GNN_LOSSES_CROSS_ENTROPY_HPP

#include "loss.hpp"

namespace gnnmath {
namespace gnn {

/**
 * @brief Cross-entropy loss function.
 *
 * Computes: CE = -(1/n) * sum(target * log(predicted))
 * Used for classification tasks. Expects predicted values to be probabilities
 * (e.g., after softmax).
 */
class cross_entropy_loss : public loss_function {
public:
    double compute(const std::vector<feature_vec>& predicted,
                  const std::vector<feature_vec>& target) const override;

    std::vector<feature_vec> gradient(const std::vector<feature_vec>& predicted,
                                      const std::vector<feature_vec>& target) const override;

    std::string name() const override { return "cross_entropy"; }
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_LOSSES_CROSS_ENTROPY_HPP
