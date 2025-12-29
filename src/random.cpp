#include <gnnmath/core/random.hpp>
#include <limits>
#include <stdexcept>
#include <thread>
#include <cmath>

namespace gnnmath {
namespace random {
    // Thread-local random engine for thread-safety
    thread_local std::mt19937 engine(std::random_device{}());

    /*
        static thread_local std::mt19937 engine(std::random_device{}());
    */

    void seed(index_t s) {
        engine.seed(static_cast<std::mt19937::result_type>(s));
    }

    scalar_t uniform(scalar_t min, scalar_t max) {
        if (min > max) {
            throw std::invalid_argument("uniform: min must be less than or equal to max");
        }
        if (!std::isfinite(min) || !std::isfinite(max)) {
            throw std::invalid_argument("uniform: min and max must be finite");
        }

        std::uniform_real_distribution<scalar_t> dist(min, max);
        scalar_t value = dist(engine);
        if (!std::isfinite(value)) {
            throw std::runtime_error("uniform: generated non-finite value");
        }

        return value;
    }
    
    std::vector<scalar_t> uniform_vector(index_t n, scalar_t min, scalar_t max) {
        if (n == 0) {
            throw std::invalid_argument("uniform_vector: size must be non-zero");
        }
        if (min > max) {
            throw std::invalid_argument("uniform_vector: min must be less than or equal to max");
        }
        if (!std::isfinite(min) || !std::isfinite(max)) {
            throw std::invalid_argument("uniform_vector: min and max must be finite");
        }

        std::vector<scalar_t> result(n);
        std::uniform_real_distribution<scalar_t> dist(min, max);
        for (index_t i = 0; i < n; ++i) {
            result[i] = dist(engine);
            if (!std::isfinite(result[i])) {
                throw std::runtime_error("uniform_vector: generated non-finite value");
            }
        }

        return result;
    }
}
}