#ifndef GNNMATH_CORE_TYPES_HPP
#define GNNMATH_CORE_TYPES_HPP

#include <cstddef>
#include <vector>

namespace gnnmath {

/// @brief Index type for array/container indexing
using index_t = std::size_t;

/// @brief Scalar type for floating point values
using scalar_t = double;

/// @brief Feature vector type
using feature_t = std::vector<scalar_t>;

/// @brief Collection of feature vectors
using feature_matrix_t = std::vector<feature_t>;

} // namespace gnnmath

#endif // GNNMATH_CORE_TYPES_HPP
