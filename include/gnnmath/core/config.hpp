#ifndef GNNMATH_CORE_CONFIG_HPP
#define GNNMATH_CORE_CONFIG_HPP

#include <cstdint>

namespace gnnmath {

/// @brief Library version information
struct version {
    static constexpr int major = 1;
    static constexpr int minor = 0;
    static constexpr int patch = 0;

    static constexpr const char* string = "1.0.0";
};

/// @brief Compile-time configuration constants
namespace config {
    /// @brief Default epsilon for floating point comparisons
    constexpr double epsilon = 1e-10;

    /// @brief Maximum value to prevent overflow in exp()
    constexpr double exp_max = 700.0;

    /// @brief Threshold for parallel execution (elements)
    constexpr std::size_t parallel_threshold = 1000;

    /// @brief Default learning rate for training
    constexpr double default_learning_rate = 0.01;

    /// @brief Adam optimizer beta1 (first moment decay)
    constexpr double adam_beta1 = 0.9;

    /// @brief Adam optimizer beta2 (second moment decay)
    constexpr double adam_beta2 = 0.999;

    /// @brief Adam optimizer epsilon (numerical stability)
    constexpr double adam_epsilon = 1e-8;
}

} // namespace gnnmath

#endif // GNNMATH_CORE_CONFIG_HPP
