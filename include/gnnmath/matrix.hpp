#ifndef GNNMATH_MATRIX_HPP
#define GNNMATH_MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <execution>
#include "vector.hpp"

namespace gnnmath {
namespace matrix {
    using vector = gnnmath::vector::vector;
    using dense_matrix = std::vector<std::vector<double>>; ///< Dense matrix type (rows x cols).

    /// @brief Compressed Sparse Row (CSR) matrix structure for sparse graph operations.
    struct CSR {
        std::vector<double> vals;      ///< Non-zero values.
        std::vector<std::size_t> col_ind; ///< Column indices of non-zero elements.
        std::vector<std::size_t> row_ptr; ///< Row start indices (last element is total non-zeros).
        std::size_t rows = 0;          ///< Number of rows.
        std::size_t cols = 0;          ///< Number of columns.

        /// @brief Constructs an empty CSR matrix with given dimensions.
        /// @param r Number of rows.
        /// @param c Number of columns.
        /// @throws std::runtime_error If dimensions are invalid (e.g., zero).
        CSR(std::size_t r, std::size_t c);

        /// @brief Converts a dense matrix to CSR format.
        /// @param rhs Input dense matrix.
        /// @throws std::runtime_error If dense matrix is invalid (e.g., inconsistent row sizes).
        CSR(const dense_matrix& rhs);

        /// @brief Multiplies the CSR matrix by a vector.
        /// @param x Input vector.
        /// @return Resulting vector.
        /// @throws std::runtime_error If matrix columns do not match vector size.
        vector multiply(const vector& x) const;
    };

    /// @brief Multiplies a dense matrix by a vector.
    /// @param matrix Input matrix (rows x cols).
    /// @param vec Input vector (cols x 1).
    /// @return Resulting vector (rows x 1).
    /// @throws std::runtime_error If matrix columns do not match vector size or matrix is invalid.
    vector matrix_vector_multiply(const dense_matrix& matrix, const vector& vec);

    /// @brief Multiplies a sparse matrix by a vector.
    /// @param matrix Input CSR matrix.
    /// @param vec Input vector.
    /// @return Resulting vector.
    /// @throws std::runtime_error If matrix columns do not match vector size.
    vector sparse_matrix_vector_multiply(const CSR& matrix, const vector& vec);

    /// @brief Multiplies two dense matrices.
    /// @param A First matrix (rows_A x cols_A).
    /// @param B Second matrix (cols_A x cols_B).
    /// @return Resulting matrix (rows_A x cols_B).
    /// @throws std::runtime_error If dimensions are incompatible or matrices are invalid.
    dense_matrix operator*(const dense_matrix& A, const dense_matrix& B);

    /// @brief Multiplies two sparse matrices.
    /// @param A First CSR matrix.
    /// @param B Second CSR matrix.
    /// @return Resulting CSR matrix.
    /// @throws std::runtime_error If dimensions are incompatible.
    CSR sparse_matrix_multiply(const CSR& A, const CSR& B);

    /// @brief Transposes a dense matrix.
    /// @param A Input matrix.
    /// @return Transposed matrix.
    /// @throws std::runtime_error If matrix is empty.
    dense_matrix transpose(const dense_matrix& A);

    /// @brief Transposes a sparse matrix.
    /// @param A Input CSR matrix.
    /// @return Transposed CSR matrix.
    /// @throws std::runtime_error If matrix is empty.
    CSR sparse_transpose(const CSR& A);

    /// @brief Adds two dense matrices element-wise.
    /// @param A First matrix.
    /// @param B Second matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions do not match.
    dense_matrix operator+(const dense_matrix& A, const dense_matrix& B);

    /// @brief Adds a dense matrix to another in-place.
    /// @param A Matrix to modify.
    /// @param B Matrix to add.
    /// @return Reference to modified matrix.
    /// @throws std::runtime_error If dimensions do not match.
    dense_matrix& operator+=(dense_matrix& A, const dense_matrix& B);

    /// @brief Subtracts two dense matrices element-wise.
    /// @param A First matrix.
    /// @param B Second matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions do not match.
    dense_matrix operator-(const dense_matrix& A, const dense_matrix& B);

    /// @brief Subtracts a dense matrix from another in-place.
    /// @param A Matrix to modify.
    /// @param B Matrix to subtract.
    /// @return Reference to modified matrix.
    /// @throws std::runtime_error If dimensions do not match.
    dense_matrix& operator-=(dense_matrix& A, const dense_matrix& B);

    /// @brief Adds two sparse matrices.
    /// @param A First CSR matrix.
    /// @param B Second CSR matrix.
    /// @return Resulting CSR matrix.
    /// @throws std::runtime_error If dimensions do not match.
    CSR operator+(const CSR& A, const CSR& B);

    /// @brief Adds a sparse matrix to another in-place.
    /// @param A CSR matrix to modify.
    /// @param B CSR matrix to add.
    /// @return Reference to modified matrix.
    /// @throws std::runtime_error If dimensions do not match.
    CSR& operator+=(CSR& A, const CSR& B);

    /// @brief Subtracts two sparse matrices.
    /// @param A First CSR matrix.
    /// @param B Second CSR matrix.
    /// @return Resulting CSR matrix.
    /// @throws std::runtime_error If dimensions do not match.
    CSR operator-(const CSR& A, const CSR& B);

    /// @brief Subtracts a sparse matrix from another in-place.
    /// @param A CSR matrix to modify.
    /// @param B CSR matrix to subtract.
    /// @return Reference to modified matrix.
    /// @throws std::runtime_error If dimensions do not match.
    CSR& operator-=(CSR& A, const CSR& B);

    /// @brief Creates an n x n identity matrix (dense).
    /// @param n Matrix dimension.
    /// @return Identity matrix.
    /// @throws std::runtime_error If n is zero.
    dense_matrix I(std::size_t n);

    /// @brief Creates an n x n identity matrix (sparse).
    /// @param n Matrix dimension.
    /// @return Identity matrix in CSR format.
    /// @throws std::runtime_error If n is zero.
    CSR Identity(std::size_t n);

    /// @brief Builds an adjacency matrix from edges.
    /// @param num_vertices Number of vertices in the graph.
    /// @param edges List of vertex pairs (edges).
    /// @return Adjacency matrix in CSR format.
    /// @throws std::runtime_error If edges are invalid.
    CSR build_adj_matrix(std::size_t num_vertices, const std::vector<std::pair<std::size_t, std::size_t>>& edges);

    /// @brief Computes the degree of each vertex.
    /// @param A Adjacency matrix in CSR format.
    /// @return Vector of vertex degrees.
    vector compute_degrees(const CSR& A);

    /// @brief Computes the Laplacian matrix (D - A).
    /// @param A Adjacency matrix in CSR format.
    /// @return Laplacian matrix in CSR format.
    CSR laplacian_matrix(const CSR& A);

    /// @brief Computes the normalized Laplacian matrix (D^(-1/2) * L * D^(-1/2)).
    /// @param A Adjacency matrix in CSR format.
    /// @return Normalized Laplacian matrix in CSR format.
    CSR normalized_laplacian_matrix(const CSR& A);

    /// @brief Validates edge list for adjacency matrix construction.
    /// @param edges List of vertex pairs.
    /// @param num_vertices Number of vertices.
    /// @return True if edges are valid, false otherwise.
    bool validate(const std::vector<std::pair<std::size_t, std::size_t>>& edges, std::size_t num_vertices);

    /// @brief Multiplies two matrices element-wise.
    /// @param A First matrix.
    /// @param B Second matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions do not match.
    dense_matrix elementwise_multiply(const dense_matrix& A, const dense_matrix& B);

    /// @brief Computes the Frobenius norm of a dense matrix.
    /// @param A Input matrix.
    /// @return Frobenius norm (sqrt(sum(A[i][j]^2))).
    double frobenius_norm(const dense_matrix& A);

    /// @brief Extracts the diagonal of a dense matrix.
    /// @param A Input matrix.
    /// @return Vector containing diagonal elements.
    /// @throws std::runtime_error If matrix is not square.
    vector extract_diagonal(const dense_matrix& A);

    /// @brief Converts a CSR matrix to dense format.
    /// @param A Input CSR matrix.
    /// @return Dense matrix.
    dense_matrix to_dense(const CSR& A);

    /// @brief Checks if a dense matrix is valid (consistent row sizes, no NaN/infinity).
    /// @param A Input matrix.
    /// @return True if valid, false otherwise.
    bool is_valid(const dense_matrix& A);

    /// @brief Checks if a sparse matrix is symmetric.
    /// @param A Input CSR matrix.
    /// @return True if symmetric, false otherwise.
    bool is_symmetric(const CSR& A);

    /// @brief Prints a dense matrix for debugging.
    /// @param A Input matrix.
    // void print_matrix(const dense_matrix& A);
}
}

#endif