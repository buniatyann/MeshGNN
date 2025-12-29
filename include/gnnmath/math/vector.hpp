#ifndef GNNMATH_MATH_VECTOR_HPP
#define GNNMATH_MATH_VECTOR_HPP

#include "../core/types.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace gnnmath {
namespace vector {

using vector = std::vector<scalar_t>;

/// @brief Computes the element-wise sum of two vectors.
/// @param a First input vector.
/// @param b Second input vector.
/// @return A new vector containing a[i] + b[i] for each index i.
/// @throws std::runtime_error If vectors have different sizes.
vector operator+(const vector& a, const vector& b);

/// @brief Adds vector b to vector a in-place.
/// @param a Vector to modify.
/// @param b Vector to add.
/// @return Reference to modified vector a.
/// @throws std::runtime_error If vectors have different sizes.
vector& operator+=(vector& a, const vector& b);

/// @brief Computes the element-wise difference of two vectors.
/// @param a First input vector.
/// @param b Second input vector.
/// @return A new vector containing a[i] - b[i] for each index i.
/// @throws std::runtime_error If vectors have different sizes.
vector operator-(const vector& a, const vector& b);

/// @brief Subtracts vector b from vector a in-place.
/// @param a Vector to modify.
/// @param b Vector to subtract.
/// @return Reference to modified vector a.
/// @throws std::runtime_error If vectors have different sizes.
vector& operator-=(vector& a, const vector& b);

/// @brief Multiplies a vector by a scalar.
/// @param a Input vector.
/// @param b Scalar multiplier.
/// @return A new vector containing a[i] * b for each index i.
vector scalar_multiply(const vector& a, scalar_t b);

/// @brief Computes the dot product of two vectors.
/// @param a First input vector.
/// @param b Second input vector.
/// @return The dot product sum(a[i] * b[i]).
/// @throws std::runtime_error If vectors have different sizes.
scalar_t dot_product(const vector& a, const vector& b);

/// @brief Computes the Euclidean (L2) norm of a vector.
/// @param a Input vector.
/// @return The square root of sum(a[i]^2).
scalar_t euclidean_norm(const vector& a);

/// @brief Applies the ReLU activation function element-wise.
/// @param a Input vector.
/// @return A new vector with max(0, a[i]) for each index i.
vector relu(const vector& a);

/// @brief Applies the sigmoid activation function element-wise.
/// @param a Input vector.
/// @return A new vector with 1 / (1 + exp(-a[i])) for each index i.
vector sigmoid(const vector& a);

/// @brief Applies the Mish activation function element-wise.
/// @param a Input vector.
/// @return A new vector with a[i] * tanh(log(1 + exp(a[i]))) for each index i.
vector mish(const vector& a);

/// @brief Applies the Softmax activation function, normalizing the vector into a probability distribution.
/// @param a Input vector.
/// @return A new vector with exp(a[i]) / sum(exp(a[j])) for each index i.
/// @throws std::runtime_error If the vector is empty.
vector softmax(const vector& a);

/// @brief Applies the Softplus activation function element-wise.
/// @param a Input vector.
/// @return A new vector with log(1 + exp(a[i])) for each index i.
vector softplus(const vector& a);

/// @brief Applies the GELU activation function element-wise.
/// @param a Input vector.
/// @return A new vector with a[i] * Phi(a[i]) for each index i, where Phi is the standard normal CDF.
vector gelu(const vector& a);

/// @brief Applies the SiLU (Sigmoid Linear Unit) activation function element-wise.
/// @param a Input vector.
/// @return A new vector with a[i] * sigmoid(a[i]) for each index i.
vector silu(const vector& a);

/// @brief Applies the Softsign activation function element-wise.
/// @param a Input vector.
/// @return A new vector with a[i] / (1 + |a[i]|) for each index i.
vector softsign(const vector& a);

} // namespace vector
} // namespace gnnmath

#endif // GNNMATH_MATH_VECTOR_HPP
