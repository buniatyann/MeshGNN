#ifndef GNNMATH_GNN_LOSSES_LOSS_HPP
#define GNNMATH_GNN_LOSSES_LOSS_HPP

#include "../../math/vector.hpp"
#include <vector>
#include <memory>
#include <string>

namespace gnnmath {
namespace gnn {

using feature_vec = gnnmath::vector::vector;

/**
 * @brief Abstract base class for loss functions.
 *
 * Loss functions compute the error between predictions and targets,
 * and provide gradients for backpropagation.
 */
class loss_function {
public:
    virtual ~loss_function() = default;

    /**
     * @brief Computes the loss value.
     * @param predicted Predicted values.
     * @param target Target values.
     * @return Loss value.
     * @throws std::runtime_error If dimensions mismatch or result is non-finite.
     */
    virtual double compute(const std::vector<feature_vec>& predicted,
                          const std::vector<feature_vec>& target) const = 0;

    /**
     * @brief Computes the gradient of the loss with respect to predictions.
     * @param predicted Predicted values.
     * @param target Target values.
     * @return Gradient with same shape as predicted.
     * @throws std::runtime_error If dimensions mismatch.
     */
    virtual std::vector<feature_vec> gradient(const std::vector<feature_vec>& predicted,
                                              const std::vector<feature_vec>& target) const = 0;

    /**
     * @brief Returns the name of the loss function.
     * @return Name string.
     */
    virtual std::string name() const = 0;
};

/**
 * @brief Factory function to create loss functions by type name.
 * @param type Loss type ("mse" or "cross_entropy").
 * @return Unique pointer to the created loss function.
 * @throws std::invalid_argument If type is unknown.
 */
std::unique_ptr<loss_function> create_loss(const std::string& type);

} // namespace gnn
} // namespace gnnmath

#endif // GNNMATH_GNN_LOSSES_LOSS_HPP
