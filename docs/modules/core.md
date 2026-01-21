# Core Module

The core module provides fundamental type definitions, configuration constants, and utility functions used throughout the MeshGNN library.

## Files

- `include/gnnmath/core/types.hpp` - Type aliases and definitions
- `include/gnnmath/core/config.hpp` - Configuration constants and version info
- `include/gnnmath/core/random.hpp` / `src/core/random.cpp` - Random number generation

---

## Types (`types.hpp`)

The types module defines the fundamental type aliases used throughout the library for consistency and easy modification.

### Type Definitions

```cpp
namespace gnnmath {
    // Floating-point type for all numerical computations
    using scalar_t = double;

    // Index type for array/container indexing
    using index_t = std::size_t;

    // Individual feature vector
    using feature_t = std::vector<scalar_t>;

    // Collection of feature vectors (matrix of features)
    using feature_matrix_t = std::vector<feature_t>;
}
```

### Usage Notes

- **`scalar_t`**: Used for all floating-point values including vertex coordinates, feature values, weights, and gradients. Currently set to `double` for precision.

- **`index_t`**: Used for all indexing operations including vertex indices, edge indices, and matrix dimensions. Set to `std::size_t` for maximum addressable range.

- **`feature_t`**: Represents a single feature vector (e.g., features for one node or one edge). This is a simple `std::vector<double>`.

- **`feature_matrix_t`**: A collection of feature vectors, typically representing all node features or all edge features in a graph.

### Design Rationale

Using type aliases allows:
1. Easy switching between `float` and `double` precision
2. Consistent typing across the codebase
3. Clear semantic meaning in function signatures
4. Potential future optimization (e.g., SIMD-friendly types)

---

## Configuration (`config.hpp`)

The configuration module defines compile-time constants and version information.

### Version Information

```cpp
namespace gnnmath::config {
    constexpr int version_major = 1;
    constexpr int version_minor = 0;
    constexpr int version_patch = 0;
}
```

### Numerical Constants

```cpp
namespace gnnmath::config {
    // Floating-point comparison threshold
    // Used for checking near-zero values in sparse matrix construction
    constexpr scalar_t epsilon = 1e-10;

    // Maximum exponent value to prevent overflow in exp()
    // exp(700) ≈ 1e304, near double max
    constexpr scalar_t exp_max = 700.0;

    // Element threshold for enabling parallel execution
    // Operations with fewer elements run sequentially
    constexpr index_t parallel_threshold = 1000;
}
```

### Adam Optimizer Constants

```cpp
namespace gnnmath::config {
    // First moment decay rate (momentum)
    constexpr scalar_t adam_beta1 = 0.9;

    // Second moment decay rate (RMSprop-like)
    constexpr scalar_t adam_beta2 = 0.999;

    // Numerical stability constant
    constexpr scalar_t adam_epsilon = 1e-8;
}
```

### Constant Descriptions

| Constant | Value | Purpose |
|----------|-------|---------|
| `epsilon` | 1e-10 | Threshold for treating values as zero |
| `exp_max` | 700.0 | Prevents `exp()` overflow |
| `parallel_threshold` | 1000 | Minimum elements for parallelization |
| `adam_beta1` | 0.9 | Adam first moment decay |
| `adam_beta2` | 0.999 | Adam second moment decay |
| `adam_epsilon` | 1e-8 | Adam numerical stability |

---

## Random Number Generation (`random.hpp` / `random.cpp`)

The random module provides thread-safe random number generation for initialization and stochastic operations.

### Implementation Details

```cpp
namespace gnnmath {
    // Thread-local Mersenne Twister engine
    // Each thread gets its own RNG state for thread safety
    thread_local std::mt19937 rng_engine;
}
```

### Functions

#### `uniform(min, max)`

Generates a single random scalar uniformly distributed in [min, max].

```cpp
scalar_t uniform(scalar_t min, scalar_t max);
```

**Parameters:**
- `min`: Lower bound (inclusive)
- `max`: Upper bound (inclusive)

**Returns:** Random value in [min, max]

**Throws:** `std::invalid_argument` if min > max or values are non-finite

**Example:**
```cpp
// Random value between -0.1 and 0.1 for weight initialization
scalar_t weight = gnnmath::uniform(-0.1, 0.1);
```

#### `uniform_vector(n, min, max)`

Generates a vector of n random scalars uniformly distributed in [min, max].

```cpp
std::vector<scalar_t> uniform_vector(index_t n, scalar_t min, scalar_t max);
```

**Parameters:**
- `n`: Number of random values to generate
- `min`: Lower bound (inclusive)
- `max`: Upper bound (inclusive)

**Returns:** Vector of n random values

**Throws:** `std::invalid_argument` if min > max, values are non-finite, or n is 0

**Example:**
```cpp
// Initialize bias vector with small random values
auto bias = gnnmath::uniform_vector(hidden_dim, -0.01, 0.01);
```

#### `seed(s)`

Sets the random seed for reproducibility.

```cpp
void seed(unsigned int s);
```

**Parameters:**
- `s`: Seed value

**Note:** This sets the seed for the current thread's RNG only. In multi-threaded code, each thread must call `seed()` separately.

**Example:**
```cpp
// Set seed for reproducible experiments
gnnmath::seed(42);

// Now all random operations on this thread are deterministic
auto weights = gnnmath::uniform_vector(100, -0.1, 0.1);
```

### Thread Safety

The random number generator uses `thread_local` storage, meaning:

1. **No Lock Contention**: Each thread has its own RNG state, eliminating synchronization overhead
2. **Independent Sequences**: Different threads generate independent random sequences
3. **Reproducibility Per Thread**: Setting seed on one thread doesn't affect others

### Error Handling

All random functions validate inputs:

```cpp
// These will throw std::invalid_argument:
uniform(5.0, 3.0);           // min > max
uniform(INFINITY, 1.0);      // non-finite bounds
uniform_vector(0, 0.0, 1.0); // n = 0
```

### Usage Patterns

#### Weight Initialization (Xavier/Glorot)

```cpp
index_t fan_in = 64;
index_t fan_out = 32;
scalar_t limit = std::sqrt(6.0 / (fan_in + fan_out));
auto weights = gnnmath::uniform_vector(fan_in * fan_out, -limit, limit);
```

#### Random Sampling

```cpp
// Select random vertices for sampling
std::vector<index_t> sample_indices;
for (index_t i = 0; i < sample_count; ++i) {
    index_t idx = static_cast<index_t>(
        gnnmath::uniform(0, static_cast<scalar_t>(num_vertices - 1))
    );
    sample_indices.push_back(idx);
}
```

#### Adding Noise to Mesh

```cpp
// Perturb vertex positions
for (auto& vertex : vertices) {
    vertex[0] += gnnmath::uniform(-noise_scale, noise_scale);
    vertex[1] += gnnmath::uniform(-noise_scale, noise_scale);
    vertex[2] += gnnmath::uniform(-noise_scale, noise_scale);
}
```

---

## Integration Example

```cpp
#include <gnnmath/core/types.hpp>
#include <gnnmath/core/config.hpp>
#include <gnnmath/core/random.hpp>

using namespace gnnmath;

int main() {
    // Set seed for reproducibility
    seed(42);

    // Use type aliases for consistency
    feature_matrix_t weights;

    // Initialize weights with bounded random values
    index_t rows = 64;
    index_t cols = 32;

    for (index_t i = 0; i < rows; ++i) {
        feature_t row = uniform_vector(cols, -0.1, 0.1);
        weights.push_back(row);
    }

    // Check values against epsilon
    for (const auto& row : weights) {
        for (scalar_t val : row) {
            if (std::abs(val) < config::epsilon) {
                // Treat as zero
            }
        }
    }

    return 0;
}
```

---

## Performance Considerations

1. **Type Sizes**: `scalar_t = double` uses 8 bytes vs 4 for float. Consider memory impact for large meshes.

2. **Threshold Tuning**: `parallel_threshold = 1000` may need adjustment based on hardware. Lower values increase parallelization but add overhead for small operations.

3. **RNG Performance**: Thread-local MT19937 is fast but uses ~2.5KB state per thread. For many short-lived threads, consider a lighter RNG.
