# Math Module

The math module provides all mathematical operations for MeshGNN, including vector arithmetic, activation functions, and both dense and sparse matrix operations.

## Files

- `include/gnnmath/math/vector.hpp` / `src/math/vector.cpp` - Vector operations and activation functions
- `include/gnnmath/math/dense_matrix.hpp` / `src/math/dense_matrix.cpp` - Dense matrix class
- `include/gnnmath/math/sparse_matrix.hpp` / `src/math/sparse_matrix.cpp` - CSR sparse matrix class

---

## Vector Operations (`vector.hpp`)

The vector module provides element-wise operations and neural network activation functions operating on `std::vector<scalar_t>`.

### Arithmetic Operations

#### `operator+(a, b)`

Element-wise vector addition.

```cpp
std::vector<scalar_t> operator+(
    const std::vector<scalar_t>& a,
    const std::vector<scalar_t>& b
);
```

**Throws:** `std::invalid_argument` if vectors have different sizes

**Complexity:** O(n)

#### `operator-(a, b)`

Element-wise vector subtraction.

```cpp
std::vector<scalar_t> operator-(
    const std::vector<scalar_t>& a,
    const std::vector<scalar_t>& b
);
```

**Throws:** `std::invalid_argument` if vectors have different sizes

#### `operator+=(a, b)` / `operator-=(a, b)`

In-place element-wise operations.

```cpp
std::vector<scalar_t>& operator+=(
    std::vector<scalar_t>& a,
    const std::vector<scalar_t>& b
);
```

#### `scalar_multiply(a, b)`

Element-wise multiplication by a scalar.

```cpp
std::vector<scalar_t> scalar_multiply(
    const std::vector<scalar_t>& a,
    scalar_t b
);
```

**Example:**
```cpp
auto scaled = gnnmath::vector::scalar_multiply(features, 0.5);
```

### Linear Algebra

#### `dot_product(a, b)`

Computes the dot product (inner product) of two vectors.

```cpp
scalar_t dot_product(
    const std::vector<scalar_t>& a,
    const std::vector<scalar_t>& b
);
```

**Formula:** `result = Σᵢ aᵢ × bᵢ`

**Throws:** `std::invalid_argument` if vectors have different sizes

**Example:**
```cpp
std::vector<scalar_t> a = {1.0, 2.0, 3.0};
std::vector<scalar_t> b = {4.0, 5.0, 6.0};
scalar_t result = gnnmath::vector::dot_product(a, b);  // 32.0
```

#### `euclidean_norm(a)`

Computes the L2 (Euclidean) norm of a vector.

```cpp
scalar_t euclidean_norm(const std::vector<scalar_t>& a);
```

**Formula:** `||a||₂ = √(Σᵢ aᵢ²)`

**Example:**
```cpp
std::vector<scalar_t> v = {3.0, 4.0};
scalar_t norm = gnnmath::vector::euclidean_norm(v);  // 5.0
```

---

### Activation Functions

All activation functions operate element-wise on vectors and use parallel execution for vectors larger than `config::parallel_threshold`.

#### `relu(a)`

Rectified Linear Unit activation.

```cpp
std::vector<scalar_t> relu(const std::vector<scalar_t>& a);
```

**Formula:** `ReLU(x) = max(0, x)`

**Properties:**
- Output range: [0, +∞)
- Non-differentiable at x=0
- Simple and fast computation

```
     │      ╱
     │    ╱
     │  ╱
─────┼──────────
     │
```

#### `sigmoid(a)`

Logistic sigmoid activation.

```cpp
std::vector<scalar_t> sigmoid(const std::vector<scalar_t>& a);
```

**Formula:** `σ(x) = 1 / (1 + e⁻ˣ)`

**Implementation Details:**
- Clamps input to [-36.7, 36.7] to prevent overflow
- Output range: (0, 1)
- Smooth, differentiable everywhere

```
    1│     ────────
     │   ╱
 0.5│──╱───────────
     │╱
    0│────────
```

#### `mish(a)`

Mish activation function (self-regularized non-monotonic).

```cpp
std::vector<scalar_t> mish(const std::vector<scalar_t>& a);
```

**Formula:** `Mish(x) = x × tanh(softplus(x)) = x × tanh(ln(1 + eˣ))`

**Properties:**
- Smooth, non-monotonic
- Unbounded above, bounded below (~-0.31)
- Self-regularizing through non-monotonicity

#### `softmax(a)`

Softmax activation for probability distributions.

```cpp
std::vector<scalar_t> softmax(const std::vector<scalar_t>& a);
```

**Formula:** `softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ`

**Implementation Details:**
- Subtracts max value before exp() for numerical stability
- Output sums to 1.0
- Used for classification outputs

**Example:**
```cpp
std::vector<scalar_t> logits = {2.0, 1.0, 0.1};
auto probs = gnnmath::vector::softmax(logits);
// probs ≈ [0.659, 0.242, 0.099], sum = 1.0
```

#### `softplus(a)`

Smooth approximation to ReLU.

```cpp
std::vector<scalar_t> softplus(const std::vector<scalar_t>& a);
```

**Formula:** `softplus(x) = ln(1 + eˣ)`

**Properties:**
- Smooth version of ReLU
- Always positive
- Derivative is sigmoid

#### `gelu(a)`

Gaussian Error Linear Unit.

```cpp
std::vector<scalar_t> gelu(const std::vector<scalar_t>& a);
```

**Formula:** `GELU(x) = x × Φ(x) = x × 0.5 × (1 + erf(x/√2))`

**Properties:**
- Smooth, non-monotonic
- Used in Transformers (BERT, GPT)
- Probabilistic interpretation

#### `silu(a)`

Sigmoid Linear Unit (also called Swish).

```cpp
std::vector<scalar_t> silu(const std::vector<scalar_t>& a);
```

**Formula:** `SiLU(x) = x × σ(x)`

**Properties:**
- Smooth, non-monotonic
- Self-gated activation
- Similar to Mish but simpler

#### `softsign(a)`

Softsign activation (similar to tanh but with slower saturation).

```cpp
std::vector<scalar_t> softsign(const std::vector<scalar_t>& a);
```

**Formula:** `softsign(x) = x / (1 + |x|)`

**Properties:**
- Output range: (-1, 1)
- Slower saturation than tanh
- Polynomial computation (no exp)

### Activation Function Comparison

| Function | Range | Smooth | Monotonic | Saturates | Use Case |
|----------|-------|--------|-----------|-----------|----------|
| ReLU | [0, ∞) | No | Yes | No | Hidden layers (default) |
| Sigmoid | (0, 1) | Yes | Yes | Yes | Binary classification |
| Mish | [-0.31, ∞) | Yes | No | No | Modern deep networks |
| Softmax | (0, 1) | Yes | - | - | Multi-class output |
| GELU | ~ | Yes | No | No | Transformers |
| SiLU | ~ | Yes | No | No | Modern architectures |

---

## Dense Matrix (`dense_matrix.hpp`)

The `dense_matrix` class provides a general-purpose 2D matrix with row-major storage.

### Class Definition

```cpp
namespace gnnmath::matrix {
    class dense_matrix {
    private:
        std::vector<scalar_t> data_;  // Flat row-major storage
        index_t rows_;
        index_t cols_;

    public:
        dense_matrix(index_t rows, index_t cols);
        dense_matrix(index_t rows, index_t cols,
                     const std::vector<scalar_t>& data);

        // Element access
        scalar_t& operator()(index_t i, index_t j);
        scalar_t operator()(index_t i, index_t j) const;

        // Dimensions
        index_t rows() const;
        index_t cols() const;

        // Raw data access
        const std::vector<scalar_t>& data() const;
        std::vector<scalar_t>& data();
    };
}
```

### Storage Format

The matrix uses **row-major** flat storage for cache efficiency:

```
Matrix:     Storage:
┌─────────┐
│ a b c d │  data_ = [a, b, c, d, e, f, g, h, i, j, k, l]
│ e f g h │
│ i j k l │  Index(i,j) = i * cols + j
└─────────┘
```

### Matrix Operations

#### Matrix-Vector Multiplication

```cpp
std::vector<scalar_t> matrix_vector_multiply(
    const dense_matrix& A,
    const std::vector<scalar_t>& v
);
```

**Formula:** `(Av)ᵢ = Σⱼ Aᵢⱼ × vⱼ`

**Complexity:** O(rows × cols)

**Throws:** `std::invalid_argument` if dimensions don't match

#### Matrix Multiplication

```cpp
dense_matrix operator*(const dense_matrix& A, const dense_matrix& B);
```

**Formula:** `(AB)ᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ`

**Complexity:** O(m × n × p) for (m×n) × (n×p)

**Throws:** `std::invalid_argument` if A.cols() ≠ B.rows()

#### Transpose

```cpp
dense_matrix transpose(const dense_matrix& A);
```

**Returns:** Aᵀ where (Aᵀ)ᵢⱼ = Aⱼᵢ

**Complexity:** O(rows × cols)

#### Element-wise Operations

```cpp
dense_matrix operator+(const dense_matrix& A, const dense_matrix& B);
dense_matrix operator-(const dense_matrix& A, const dense_matrix& B);
dense_matrix elementwise_multiply(const dense_matrix& A, const dense_matrix& B);
```

All require matching dimensions.

#### Utility Functions

```cpp
// Create n×n identity matrix
dense_matrix I(index_t n);

// Frobenius norm: ||A||_F = √(Σᵢⱼ Aᵢⱼ²)
scalar_t frobenius_norm(const dense_matrix& A);

// Extract diagonal from square matrix
std::vector<scalar_t> extract_diagonal(const dense_matrix& A);

// Check all values are finite
bool is_valid(const dense_matrix& A);
```

### Example Usage

```cpp
using namespace gnnmath::matrix;

// Create 3x3 matrix
dense_matrix A(3, 3);
A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;

// Matrix-vector product
std::vector<scalar_t> v = {1.0, 0.0, 0.0};
auto result = matrix_vector_multiply(A, v);  // [1.0, 4.0, 7.0]

// Matrix multiplication
dense_matrix B = transpose(A);
dense_matrix C = A * B;  // AAᵀ

// Compute norm
scalar_t norm = frobenius_norm(C);
```

---

## Sparse Matrix - CSR Format (`sparse_matrix.hpp`)

The `sparse_matrix` class implements the **Compressed Sparse Row (CSR)** format for efficient sparse matrix operations, crucial for graph adjacency matrices.

### CSR Format Explanation

CSR uses three arrays to store only non-zero values:

```
Dense Matrix (5x5):           CSR Representation:
┌─────────────────┐
│ 1 0 0 2 0 │           vals     = [1, 2, 3, 4, 5, 6]
│ 0 3 0 0 0 │           col_ind  = [0, 3, 1, 2, 4, 0]
│ 0 0 4 0 5 │           row_ptr  = [0, 2, 3, 5, 5, 6]
│ 0 0 0 0 0 │                      ↑  ↑  ↑  ↑  ↑  ↑
│ 6 0 0 0 0 │                      │  │  │  │  │  └─ end
└─────────────────┘                 │  │  │  │  └─ row 4 starts at index 5
                                    │  │  │  └─ row 3 starts at index 5 (empty)
                                    │  │  └─ row 2 starts at index 3
                                    │  └─ row 1 starts at index 2
                                    └─ row 0 starts at index 0
```

**Structure:**
- `vals[k]` = k-th non-zero value (row-by-row)
- `col_ind[k]` = column index of k-th non-zero value
- `row_ptr[i]` = index into vals/col_ind where row i starts
- `row_ptr[rows]` = total number of non-zeros (nnz)

### Class Definition

```cpp
namespace gnnmath::matrix {
    class sparse_matrix {
    private:
        std::vector<scalar_t> vals_;
        std::vector<index_t> col_ind_;
        std::vector<index_t> row_ptr_;
        index_t rows_;
        index_t cols_;

    public:
        // Constructors
        sparse_matrix(index_t rows, index_t cols);  // Empty matrix
        sparse_matrix(const dense_matrix& dense);   // Convert from dense
        sparse_matrix(index_t rows, index_t cols,
                      std::vector<scalar_t>&& vals,
                      std::vector<index_t>&& col_ind,
                      std::vector<index_t>&& row_ptr);

        // Accessors
        index_t rows() const;
        index_t cols() const;
        index_t nnz() const;  // Number of non-zeros

        const std::vector<scalar_t>& vals() const;
        const std::vector<index_t>& col_ind() const;
        const std::vector<index_t>& row_ptr() const;

        // Operations
        std::vector<scalar_t> multiply(const std::vector<scalar_t>& x) const;
        bool validate() const;
    };
}
```

### Sparse Matrix Operations

#### Sparse Matrix-Vector Multiplication

```cpp
std::vector<scalar_t> sparse_matrix::multiply(
    const std::vector<scalar_t>& x
) const;
```

**Algorithm:**
```cpp
for each row i:
    result[i] = 0
    for k = row_ptr[i] to row_ptr[i+1]:
        result[i] += vals[k] * x[col_ind[k]]
```

**Complexity:** O(nnz) - only touches non-zero elements

**Throws:** `std::invalid_argument` if x.size() ≠ cols

#### Sparse Matrix Addition/Subtraction

```cpp
sparse_matrix operator+(const sparse_matrix& A, const sparse_matrix& B);
sparse_matrix operator-(const sparse_matrix& A, const sparse_matrix& B);
```

**Complexity:** O(nnz_A + nnz_B)

#### Sparse Matrix Multiplication

```cpp
sparse_matrix sparse_matrix_multiply(
    const sparse_matrix& A,
    const sparse_matrix& B
);
```

**Complexity:** O(nnz_A × avg_nnz_per_row_B)

#### Sparse Transpose

```cpp
sparse_matrix sparse_transpose(const sparse_matrix& A);
```

**Returns:** Aᵀ in CSR format

**Complexity:** O(nnz + rows + cols)

### Graph-Specific Operations

#### Build Adjacency Matrix

```cpp
sparse_matrix build_adj_matrix(
    index_t num_vertices,
    const std::vector<std::pair<index_t, index_t>>& edges
);
```

Creates a symmetric adjacency matrix from an edge list.

**Example:**
```cpp
std::vector<std::pair<index_t, index_t>> edges = {
    {0, 1}, {1, 2}, {0, 2}
};
auto adj = build_adj_matrix(3, edges);
// Creates:
// [0 1 1]
// [1 0 1]
// [1 1 0]
```

#### Compute Vertex Degrees

```cpp
std::vector<scalar_t> compute_degrees(const sparse_matrix& A);
```

**Returns:** Vector where degrees[i] = sum of row i (number of neighbors)

#### Laplacian Matrix

```cpp
sparse_matrix laplacian_matrix(const sparse_matrix& A);
```

**Formula:** `L = D - A`

Where D is the diagonal degree matrix.

**Properties:**
- Symmetric positive semi-definite
- Eigenvalues reveal graph connectivity
- L1 = 0 (constant eigenvector)

#### Normalized Laplacian

```cpp
sparse_matrix normalized_laplacian_matrix(const sparse_matrix& A);
```

**Formula:** `L_norm = D^(-1/2) × L × D^(-1/2) = I - D^(-1/2) × A × D^(-1/2)`

**Properties:**
- Eigenvalues in [0, 2]
- Better conditioned for GNN training
- Used in spectral graph convolutions

### Utility Functions

```cpp
// Sparse identity matrix
sparse_matrix Identity(index_t n);

// Check symmetry (A = Aᵀ)
bool is_symmetric(const sparse_matrix& A);

// Convert to dense matrix
dense_matrix to_dense(const sparse_matrix& A);

// Validate CSR structure
bool sparse_matrix::validate() const;
```

### Validation

The `validate()` method checks:
1. `row_ptr` has size `rows + 1`
2. `row_ptr[0] == 0` and `row_ptr[rows] == nnz`
3. `row_ptr` is monotonically non-decreasing
4. All `col_ind` values are < `cols`
5. Column indices within each row are sorted
6. All values are finite (no NaN/Inf)

### Example Usage

```cpp
using namespace gnnmath::matrix;

// Build adjacency from edge list
std::vector<std::pair<index_t, index_t>> edges = {
    {0, 1}, {0, 2}, {1, 2}, {2, 3}
};
auto A = build_adj_matrix(4, edges);

// Compute graph Laplacian for spectral analysis
auto L = laplacian_matrix(A);
auto L_norm = normalized_laplacian_matrix(A);

// Sparse matrix-vector multiplication (message passing)
std::vector<scalar_t> node_values = {1.0, 2.0, 3.0, 4.0};
auto aggregated = A.multiply(node_values);
// aggregated[i] = sum of neighbor values for node i

// Check structure
if (!A.validate()) {
    throw std::runtime_error("Invalid CSR structure");
}
```

---

## Memory and Performance Considerations

### Dense vs Sparse Trade-offs

| Aspect | Dense | Sparse (CSR) |
|--------|-------|--------------|
| Storage | O(n²) | O(nnz + n) |
| Element access | O(1) | O(log(nnz/n)) |
| Matrix-vector | O(n²) | O(nnz) |
| Matrix multiply | O(n³) | O(nnz × avg_row) |
| Best for | Small, dense matrices | Large, sparse graphs |

### When to Use Sparse

Use sparse matrices when:
- Density < 10-15% (graph adjacency typically < 1%)
- Matrix dimension > 1000
- Most operations are matrix-vector products

### Parallel Execution

Vector operations automatically parallelize when:
- Vector size > `config::parallel_threshold` (default 1000)
- Compiled with parallel execution support (TBB)

```cpp
// Automatically parallel for large vectors
auto result = gnnmath::vector::relu(large_features);
```

### Cache Efficiency

- Dense matrices use row-major storage for row-wise access patterns
- CSR format provides good cache locality for row iteration
- Feature matrices (vector of vectors) may have worse locality than flat storage
