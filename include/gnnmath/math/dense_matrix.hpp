#ifndef GNNMATH_MATH_DENSE_MATRIX_HPP
#define GNNMATH_MATH_DENSE_MATRIX_HPP

#include "../core/types.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace gnnmath {
namespace matrix {

using vector = std::vector<scalar_t>;

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
    dense_matrix(const std::vector<std::vector<scalar_t>>& data);

    /// @brief Returns number of rows.
    std::size_t rows() const { return rows_; }

    /// @brief Returns number of columns.
    std::size_t cols() const { return cols_; }

    /// @brief Const element access.
    /// @param i Row index.
    /// @param j Column index.
    /// @return Element value.
    /// @throws std::out_of_range If indices are invalid.
    scalar_t operator()(std::size_t i, std::size_t j) const;

    /// @brief Non-const element access.
    /// @param i Row index.
    /// @param j Column index.
    /// @return Element value.
    /// @throws std::out_of_range If indices are invalid.
    scalar_t& operator()(std::size_t i, std::size_t j);

    /// @brief Returns raw data.
    const std::vector<scalar_t>& data() const { return data_; }

private:
    std::vector<scalar_t> data_; ///< Flat storage (row-major).
    std::size_t rows_, cols_;    ///< Dimensions.
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

/// @brief Transposes a dense matrix.
/// @param A Input matrix.
/// @return Transposed matrix.
/// @throws std::runtime_error If matrix is empty.
dense_matrix transpose(const dense_matrix& A);

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

/// @brief Checks if a dense matrix is valid.
/// @param A Input matrix.
/// @return True if valid.
bool is_valid(const dense_matrix& A);

} // namespace matrix
} // namespace gnnmath

#endif // GNNMATH_MATH_DENSE_MATRIX_HPP
