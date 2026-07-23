#include <gnnmath/math/dense_matrix.hpp>
#include <cmath>
#include <stdexcept>

namespace gnnmath {
namespace matrix {

dense_matrix::dense_matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols) {
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("dense_matrix: dimensions must be non-zero");
    }

    data_.resize(rows * cols, 0.0);
}

dense_matrix::dense_matrix(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        throw std::runtime_error("dense_matrix: empty input");
    }

    rows_ = data.size();
    cols_ = data[0].size();
    data_.reserve(rows_ * cols_);
    for (const auto& row : data) {
        if (row.size() != cols_) {
            throw std::runtime_error("dense_matrix: inconsistent row sizes");
        }

        for (double x : row) {
            if (!std::isfinite(x)) {
                throw std::runtime_error("dense_matrix: non-finite value");
            }

            data_.push_back(x);
        }
    }
}

double dense_matrix::operator()(std::size_t i, std::size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("dense_matrix: index out of bounds");
    }

    return data_[i * cols_ + j];
}

double& dense_matrix::operator()(std::size_t i, std::size_t j) {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("dense_matrix: index out of bounds");
    }

    return data_[i * cols_ + j];
}

vector matrix_vector_multiply(const dense_matrix& matrix, const vector& vec) {
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        throw std::runtime_error("matrix_vector_multiply: empty matrix");
    }

    if (matrix.cols() != vec.size()) {
        throw std::runtime_error("matrix_vector_multiply: dimension mismatch");
    }

    vector ans(matrix.rows(), 0.0);
    const scalar_t* a = matrix.raw();
    const std::size_t cols = matrix.cols();
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        const scalar_t* row = a + i * cols;
        double sum = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            sum += row[j] * vec[j];
        }

        if (!std::isfinite(sum)) {
            throw std::runtime_error("matrix_vector_multiply: non-finite result");
        }

        ans[i] = sum;
    }

    return ans;
}

dense_matrix operator*(const dense_matrix& A, const dense_matrix& B) {
    if (A.rows() == 0 || A.cols() == 0 || B.rows() == 0 || B.cols() == 0) {
        throw std::runtime_error("matrix_multiply: empty matrix");
    }
    if (A.cols() != B.rows()) {
        throw std::runtime_error("matrix_multiply: dimension mismatch");
    }

    const std::size_t n = A.rows(), k_dim = A.cols(), m = B.cols();
    dense_matrix C(n, m);
    const scalar_t* a = A.raw();
    const scalar_t* b = B.raw();
    scalar_t* c = C.raw();
    // i-k-j order: both B and C rows are traversed contiguously
    for (std::size_t i = 0; i < n; ++i) {
        scalar_t* c_row = c + i * m;
        for (std::size_t k = 0; k < k_dim; ++k) {
            double a_ik = a[i * k_dim + k];
            const scalar_t* b_row = b + k * m;
            for (std::size_t j = 0; j < m; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }

        for (std::size_t j = 0; j < m; ++j) {
            if (!std::isfinite(c_row[j])) {
                throw std::runtime_error("matrix_multiply: non-finite result");
            }
        }
    }

    return C;
}

dense_matrix transpose(const dense_matrix& A) {
    if (A.rows() == 0 || A.cols() == 0) {
        throw std::runtime_error("transpose: empty matrix");
    }

    dense_matrix B(A.cols(), A.rows());
    const scalar_t* a = A.raw();
    scalar_t* b = B.raw();
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            b[j * A.rows() + i] = a[i * A.cols() + j];
        }
    }

    return B;
}

dense_matrix operator+(const dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("operator+: dimension mismatch");
    }

    dense_matrix C(A.rows(), A.cols());
    const scalar_t* a = A.raw();
    const scalar_t* b = B.raw();
    scalar_t* c = C.raw();
    const std::size_t total = A.rows() * A.cols();
    for (std::size_t i = 0; i < total; ++i) {
        c[i] = a[i] + b[i];
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(c[i])) {
            throw std::runtime_error("operator+: non-finite result");
        }
    }

    return C;
}

dense_matrix& operator+=(dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("operator+=: dimension mismatch");
    }

    scalar_t* a = A.raw();
    const scalar_t* b = B.raw();
    const std::size_t total = A.rows() * A.cols();
    for (std::size_t i = 0; i < total; ++i) {
        a[i] += b[i];
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(a[i])) {
            throw std::runtime_error("operator+=: non-finite result");
        }
    }

    return A;
}

dense_matrix operator-(const dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("operator-: dimension mismatch");
    }

    dense_matrix C(A.rows(), A.cols());
    const scalar_t* a = A.raw();
    const scalar_t* b = B.raw();
    scalar_t* c = C.raw();
    const std::size_t total = A.rows() * A.cols();
    for (std::size_t i = 0; i < total; ++i) {
        c[i] = a[i] - b[i];
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(c[i])) {
            throw std::runtime_error("operator-: non-finite result");
        }
    }

    return C;
}

dense_matrix& operator-=(dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("operator-=: dimension mismatch");
    }

    scalar_t* a = A.raw();
    const scalar_t* b = B.raw();
    const std::size_t total = A.rows() * A.cols();
    for (std::size_t i = 0; i < total; ++i) {
        a[i] -= b[i];
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(a[i])) {
            throw std::runtime_error("operator-=: non-finite result");
        }
    }

    return A;
}

dense_matrix I(std::size_t n) {
    if (n == 0) {
        throw std::runtime_error("I: dimension must be non-zero");
    }

    dense_matrix identity(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        identity(i, i) = 1.0;
    }

    return identity;
}

dense_matrix elementwise_multiply(const dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("elementwise_multiply: dimension mismatch");
    }

    dense_matrix C(A.rows(), A.cols());
    const scalar_t* a = A.raw();
    const scalar_t* b = B.raw();
    scalar_t* c = C.raw();
    const std::size_t total = A.rows() * A.cols();
    for (std::size_t i = 0; i < total; ++i) {
        c[i] = a[i] * b[i];
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (!std::isfinite(c[i])) {
            throw std::runtime_error("elementwise_multiply: non-finite result");
        }
    }

    return C;
}

double frobenius_norm(const dense_matrix& A) {
    if (A.rows() == 0 || A.cols() == 0) {
        throw std::runtime_error("frobenius_norm: empty matrix");
    }

    double sum = 0.0;
    const scalar_t* a = A.raw();
    const std::size_t total = A.rows() * A.cols();
    for (std::size_t i = 0; i < total; ++i) {
        sum += a[i] * a[i];
    }

    if (!std::isfinite(sum)) {
        throw std::runtime_error("frobenius_norm: non-finite result");
    }

    return std::sqrt(sum);
}

vector extract_diagonal(const dense_matrix& A) {
    if (A.rows() == 0 || A.cols() == 0) {
        throw std::runtime_error("extract_diagonal: empty matrix");
    }
    if (A.rows() != A.cols()) {
        throw std::runtime_error("extract_diagonal: matrix must be square");
    }

    vector diag(A.rows());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        diag[i] = A(i, i);
    }

    return diag;
}

bool is_valid(const dense_matrix& A) {
    if (A.rows() == 0 || A.cols() == 0) {
        return true;
    }

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            if (!std::isfinite(A(i, j))) {
                return false;
            }
        }
    }

    return true;
}

} // namespace matrix
} // namespace gnnmath
