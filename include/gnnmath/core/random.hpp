#ifndef GNNMATH_CORE_RANDOM_HPP
#define GNNMATH_CORE_RANDOM_HPP

#include "types.hpp"
#include <vector>
#include <random>

namespace gnnmath {
namespace random {

/// @brief Generates a uniform random number in [min, max].
/// @param min Lower bound (inclusive).
/// @param max Upper bound (inclusive).
/// @return Random scalar_t value.
/// @throws std::invalid_argument If min > max.
scalar_t uniform(scalar_t min, scalar_t max);

/// @brief Sets the seed for the random number generator.
/// @param s Seed value.
void seed(index_t s);

/// @brief Generates a vector of uniform random numbers.
/// @param n Number of elements.
/// @param min Lower bound (inclusive).
/// @param max Upper bound (inclusive).
/// @return Vector of random scalar_t values.
/// @throws std::invalid_argument If min > max or n is zero.
std::vector<scalar_t> uniform_vector(index_t n, scalar_t min, scalar_t max);

} // namespace random
} // namespace gnnmath

#endif // GNNMATH_CORE_RANDOM_HPP
