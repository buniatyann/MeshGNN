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

    /// @brief Dense matrix with flat storage for cache efficiency.
    class dense_matrix {
    public:
        /// @brief Constructs a dense matrix with given dimensions.
        /// @param r Number of rows.
        /// @param c Number of columns.
        /// @throws std::runtime_error If dimensions are invalid.
        dense_matrix(std::size_t r, std::size_t c);

        /// @brief Constructs from a 2D vector.
        /// @param data Input 2D vector.
        /// @throws std::runtime_error If data is invalid.
        dense_matrix(const std::vector<std::vector<double>>& data);

        /// @brief Returns number of rows.
        std::size_t rows() const { return rows_; }

        /// @brief Returns number of columns.
        std::size_t cols() const { return cols_; }

        /// @brief Const element access.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element value.
        /// @throws std::out_of_range If indices are invalid.
        double operator()(std::size_t i, std::size_t j) const;

        /// @brief Non-const element access.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element value.
        /// @throws std::out_of_range If indices are invalid.
        double& operator()(std::size_t i, std::size_t j);

        /// @brief Returns raw data.
        const std::vector<double>& data() const { return data_; }

    private:
        std::vector<double> data_; ///< Flat storage (row-major).
        std::size_t rows_, cols_;  ///< Dimensions.
    };

    /// @brief Compressed Sparse Row (CSR) matrix for sparse graph operations.
    struct sparse_matrix {
        std::vector<double> vals;      ///< Non-zero values.
        std::vector<std::size_t> col_ind; ///< Column indices of non-zero elements.
        std::vector<std::size_t> row_ptr; ///< Row start indices (last element is total non-zeros).
        std::size_t rows = 0;          ///< Number of rows.
        std::size_t cols = 0;          ///< Number of columns.

        /// @brief Constructs an empty CSR matrix.
        /// @param r Number of rows.
        /// @param c Number of columns.
        /// @throws std::runtime_error If dimensions are invalid.
        sparse_matrix(std::size_t r, std::size_t c);

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
        sparse_matrix(std::size_t r, std::size_t c,
                      std::vector<double>&& values,
                      std::vector<std::size_t>&& col_indices,
                      std::vector<std::size_t>&& row_ptrs);

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

    // private:
        /// @brief Validates CSR format.
        /// @throws std::runtime_error If invalid.
        void validate() const;
    };

    /// @brief Multiplies a dense matrix by a vector.
    /// @param matrix Input matrix.
    /// @param vec Input vector.
    /// @return Resulting vector.
    /// @throws std::runtime_error If dimensions mismatch or matrix is invalid.
    vector matrix_vector_multiply(const dense_matrix& matrix, const vector& vec);

    /// @brief Multiplies two dense matrices.
    /// @param A First matrix.
    /// @param B Second matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions are incompatible.
    dense_matrix operator*(const dense_matrix& A, const dense_matrix& B);

    /// @brief Multiplies two sparse matrices.
    /// @param A First matrix.
    /// @param B Second matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions are incompatible.
    sparse_matrix sparse_matrix_multiply(const sparse_matrix& A, const sparse_matrix& B);

    /// @brief Transposes a dense matrix.
    /// @param A Input matrix.
    /// @return Transposed matrix.
    /// @throws std::runtime_error If matrix is empty.
    dense_matrix transpose(const dense_matrix& A);

    /// @brief Transposes a sparse matrix.
    /// @param A Input matrix.
    /// @return Transposed matrix.
    /// @throws std::runtime_error If matrix is empty.
    sparse_matrix sparse_transpose(const sparse_matrix& A);

    /// @brief Adds two dense matrices element-wise.
    /// @param A First matrix.
    /// @param B Second matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions mismatch.
    dense_matrix operator+(const dense_matrix& A, const dense_matrix& B);

    /// @brief Adds a dense matrix in-place.
    /// @param A Matrix to modify.
    /// @param B Matrix to add.
    /// @return Reference to modified matrix.
    /// @throws std::runtime_error If dimensions mismatch.
    dense_matrix& operator+=(dense_matrix& A, const dense_matrix& B);

    /// @brief Subtracts two dense matrices element-wise.
    /// @param A First matrix.
    /// @param B Second matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions mismatch.
    dense_matrix operator-(const dense_matrix& A, const dense_matrix& B);

    /// @brief Subtracts a dense matrix in-place.
    /// @param A Matrix to modify.
    /// @param B Matrix to subtract.
    /// @return Reference to modified matrix.
    /// @throws std::runtime_error If dimensions mismatch.
    dense_matrix& operator-=(dense_matrix& A, const dense_matrix& B);

    /// @brief Creates an n x n identity matrix (dense).
    /// @param n Matrix dimension.
    /// @return Identity matrix.
    /// @throws std::runtime_error If n is zero.
    dense_matrix I(std::size_t n);

    /// @brief Creates an n x n identity matrix (sparse).
    /// @param n Matrix dimension.
    /// @return Identity matrix in CSR format.
    /// @throws std::runtime_error If n is zero.
    sparse_matrix Identity(std::size_t n);

    /// @brief Builds an adjacency matrix from edges.
    /// @param num_vertices Number of vertices.
    /// @param edges List of vertex pairs.
    /// @return Adjacency matrix in CSR format.
    /// @throws std::runtime_error If edges are invalid.
    sparse_matrix build_adj_matrix(std::size_t num_vertices, const std::vector<std::pair<std::size_t, std::size_t>>& edges);

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
    bool validate(const std::vector<std::pair<std::size_t, std::size_t>>& edges, std::size_t num_vertices);

    /// @brief Multiplies two dense matrices element-wise.
    /// @param A First matrix.
    /// @param B Second matrix.
    /// @return Resulting matrix.
    /// @throws std::runtime_error If dimensions mismatch.
    dense_matrix elementwise_multiply(const dense_matrix& A, const dense_matrix& B);

    /// @brief Computes Frobenius norm of a dense matrix.
    /// @param A Input matrix.
    /// @return Frobenius norm.
    double frobenius_norm(const dense_matrix& A);

    /// @brief Extracts diagonal of a dense matrix.
    /// @param A Input matrix.
    /// @return Diagonal elements.
    /// @throws std::runtime_error If not square.
    vector extract_diagonal(const dense_matrix& A);

    /// @brief Converts CSR to dense format.
    /// @param A Input CSR matrix.
    /// @return Dense matrix.
    dense_matrix to_dense(const sparse_matrix& A);

    /// @brief Checks if a dense matrix is valid.
    /// @param A Input matrix.
    /// @return True if valid.
    bool is_valid(const dense_matrix& A);

    /// @brief Checks if a sparse matrix is symmetric.
    /// @param A Input matrix.
    /// @return True if symmetric.
    bool is_symmetric(const sparse_matrix& A);
}
}

#endif
