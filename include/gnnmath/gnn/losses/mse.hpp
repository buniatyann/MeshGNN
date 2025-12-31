#ifndef GNNMATH_GNN_LOSSES_MSE_HPP
#define GNNMATH_GNN_LOSSES_MSE_HPP

#include "loss.hpp"

namespace gnnmath {
namespace gnn {

/**
 * @brief Mean Squared Error loss function.
 *
 * Computes: MSE = (1/n) * sum((predicted - target)^2)
 * Used for regression tasks.
 */
class mse_loss : public loss_function {
public:
    double compute(const std::vector<feature_vec>& predicted,
                  const std::vector<feature_vec>& target) const override;

    std::vector<feature_vec> gradient(const std::vector<feature_vec>& predicted,
                                      const std::vector<feature_vec>& target) const override;

    std::string name() const override { return "mse"; }
};

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_LOSSES_MSE_HPP
