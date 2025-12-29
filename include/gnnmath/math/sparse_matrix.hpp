#ifndef GNNMATH_MATH_SPARSE_MATRIX_HPP
#define GNNMATH_MATH_SPARSE_MATRIX_HPP

#include "../core/types.hpp"
#include "dense_matrix.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <utility>

namespace gnnmath {
namespace matrix {

/// @brief Compressed Sparse Row (CSR) matrix for sparse graph operations.
struct sparse_matrix {
    std::vector<scalar_t> vals;        ///< Non-zero values.
    std::vector<index_t> col_ind;      ///< Column indices of non-zero elements.
    std::vector<index_t> row_ptr;      ///< Row start indices (last element is total non-zeros).
    index_t rows = 0;                  ///< Number of rows.
    index_t cols = 0;                  ///< Number of columns.

    /// @brief Constructs an empty CSR matrix.
    /// @param r Number of rows.
    /// @param c Number of columns.
    /// @throws std::runtime_error If dimensions are invalid.
    sparse_matrix(index_t r, index_t c);

    /// @brief Converts a dense matrix to CSR format.
    /// @param rhs Input dense matrix.
    /// @throws std::runtime_error If dense matrix is invalid.
    sparse_matrix(const dense_matrix& rhs);

    /// @brief Constructs from raw CSR components.
    /// @param r Number of rows.
    /// @param c Number of columns.
    /// @param values Non-zero values (moved).
    /// @param col_indices Column indices (moved).
    /// @param row_ptrs Row pointers (moved).
    /// @throws std::runtime_error If inputs are invalid.
    sparse_matrix(index_t r, index_t c,
                  std::vector<scalar_t>&& values,
                  std::vector<index_t>&& col_indices,
                  std::vector<index_t>&& row_ptrs);

    /// @brief Multiplies by a vector.
    /// @param x Input vector.
    /// @return Resulting vector.
    /// @throws std::runtime_error If dimensions mismatch.
    vector multiply(const vector& x) const;

    /// @brief Adds another sparse matrix.
    /// @param rhs Right-hand side matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions mismatch.
    sparse_matrix operator+(const sparse_matrix& rhs) const;

    /// @brief Adds in-place.
    /// @param rhs Right-hand side matrix.
    /// @return Reference to modified matrix.
    /// @throws std::runtime_error If dimensions mismatch.
    sparse_matrix& operator+=(const sparse_matrix& rhs);

    /// @brief Subtracts another sparse matrix.
    /// @param rhs Right-hand side matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions mismatch.
    sparse_matrix operator-(const sparse_matrix& rhs) const;

    /// @brief Subtracts in-place.
    /// @param rhs Right-hand side matrix.
    /// @return Reference to modified matrix.
    /// @throws std::runtime_error If dimensions mismatch.
    sparse_matrix& operator-=(const sparse_matrix& rhs);

    /// @brief Validates CSR format.
    /// @throws std::runtime_error If invalid.
    void validate() const;
};

/// @brief Multiplies two sparse matrices.
/// @param A First matrix.
/// @param B Second matrix.
/// @return Resulting matrix.
/// @throws std::runtime_error If dimensions are incompatible.
sparse_matrix sparse_matrix_multiply(const sparse_matrix& A, const sparse_matrix& B);

/// @brief Transposes a sparse matrix.
/// @param A Input matrix.
/// @return Transposed matrix.
/// @throws std::runtime_error If matrix is empty.
sparse_matrix sparse_transpose(const sparse_matrix& A);

/// @brief Creates an n x n identity matrix (sparse).
/// @param n Matrix dimension.
/// @return Identity matrix in CSR format.
/// @throws std::runtime_error If n is zero.
sparse_matrix Identity(index_t n);

/// @brief Builds an adjacency matrix from edges.
/// @param num_vertices Number of vertices.
/// @param edges List of vertex pairs.
/// @return Adjacency matrix in CSR format.
/// @throws std::runtime_error If edges are invalid.
sparse_matrix build_adj_matrix(index_t num_vertices,
                               const std::vector<std::pair<index_t, index_t>>& edges);

/// @brief Computes vertex degrees.
/// @param A Adjacency matrix.
/// @return Vector of degrees.
vector compute_degrees(const sparse_matrix& A);

/// @brief Computes Laplacian matrix (D - A).
/// @param A Adjacency matrix.
/// @return Laplacian matrix.
sparse_matrix laplacian_matrix(const sparse_matrix& A);

/// @brief Computes normalized Laplacian (D^(-1/2) * L * D^(-1/2)).
/// @param A Adjacency matrix.
/// @return Normalized Laplacian.
sparse_matrix normalized_laplacian_matrix(const sparse_matrix& A);

/// @brief Validates edge list.
/// @param edges List of vertex pairs.
/// @param num_vertices Number of vertices.
/// @return True if valid.
bool validate(const std::vector<std::pair<index_t, index_t>>& edges, index_t num_vertices);

/// @brief Converts CSR to dense format.
/// @param A Input CSR matrix.
/// @return Dense matrix.
dense_matrix to_dense(const sparse_matrix& A);

/// @brief Checks if a sparse matrix is symmetric.
/// @param A Input matrix.
/// @return True if symmetric.
bool is_symmetric(const sparse_matrix& A);

} // namespace matrix
} // namespace gnnmath

#endif // GNNMATH_MATH_SPARSE_MATRIX_HPP
