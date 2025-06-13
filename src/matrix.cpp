#include "../include/gnnmath/matrix.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <execution>

namespace gnnmath {
namespace matrix {

dense_matrix::dense_matrix(std::size_t r, std::size_t c) : rows_(r), cols_(c) {
    if (r == 0 || c == 0) {
        throw std::runtime_error("dense_matrix: dimensions must be non-zero");
    }

    data_.resize(r * c, 0.0);
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

sparse_matrix::sparse_matrix(std::size_t r, std::size_t c) : rows(r), cols(c) {
    if (r == 0 || c == 0) {
        throw std::runtime_error("sparse_matrix: dimensions must be non-zero");
    }

    row_ptr.resize(r + 1, 0);
}

sparse_matrix::sparse_matrix(const dense_matrix& rhs) {
    rows = rhs.rows();
    cols = rhs.cols();
    row_ptr.resize(rows + 1);
    row_ptr[0] = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            double val = rhs(i, j);
            if (std::abs(val) > 1e-10) {
                vals.push_back(val);
                col_ind.push_back(j);
            }
        }

        row_ptr[i + 1] = vals.size();
    }
}

sparse_matrix::sparse_matrix(std::size_t r, std::size_t c, std::vector<double>&& values,
                            std::vector<std::size_t>&& col_indices, std::vector<std::size_t>&& row_ptrs)
    : rows(r), cols(c), vals(std::move(values)), col_ind(std::move(col_indices)), row_ptr(std::move(row_ptrs)) {
    validate();
}

void sparse_matrix::validate() const {
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("sparse_matrix: invalid dimensions");
    }
    if (row_ptr.size() != rows + 1) {
        throw std::runtime_error("sparse_matrix: invalid row_ptr size");
    }
    if (vals.size() != col_ind.size()) {
        throw std::runtime_error("sparse_matrix: vals and col_ind size mismatch");
    }
    if (row_ptr.back() != vals.size()) {
        throw std::runtime_error("sparse_matrix: row_ptr.back() does not match vals size");
    }

    for (std::size_t i = 0; i < col_ind.size(); ++i) {
        if (col_ind[i] >= cols) {
            throw std::runtime_error("sparse_matrix: invalid column index");
        }

        if (!std::isfinite(vals[i])) {
            throw std::runtime_error("sparse_matrix: non-finite value");
        }
    }

    for (std::size_t i = 1; i < row_ptr.size(); ++i) {
        if (row_ptr[i] < row_ptr[i-1]) {
            throw std::runtime_error("sparse_matrix: row_ptr not non-decreasing");
        }
    }
}

vector sparse_matrix::multiply(const vector& x) const {
    if (cols != x.size()) {
        throw std::runtime_error("sparse_matrix multiply: dimension mismatch");
    }

    vector ans(rows, 0.0);
    auto compute = [this, &x, &ans](double& y) {
        std::size_t i = &y - ans.data();
        for (std::size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            y += vals[j] * x[col_ind[j]];
        }

        if (!std::isfinite(y)) {
            throw std::runtime_error("sparse_matrix multiply: non-finite result");
        }
    };

    if (rows > 1000 || vals.size() > 10000) {
        std::for_each(std::execution::par_unseq, ans.begin(), ans.end(), compute);
    } 
    else {
        std::for_each(ans.begin(), ans.end(), compute);
    }
    
    return ans;
}

sparse_matrix sparse_matrix::operator+(const sparse_matrix& rhs) const {
    if (rows != rhs.rows || cols != rhs.cols) {
        throw std::runtime_error("sparse_matrix operator+: dimension mismatch");
    }
    
    sparse_matrix result(rows, cols);
    result.row_ptr[0] = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        std::size_t j = row_ptr[i], k = rhs.row_ptr[i];
        while (j < row_ptr[i + 1] || k < rhs.row_ptr[i + 1]) {
            std::size_t col_j = (j < row_ptr[i + 1]) ? col_ind[j] : cols;
            std::size_t col_k = (k < rhs.row_ptr[i + 1]) ? rhs.col_ind[k] : cols;
            double val = 0.0;
            if (col_j == col_k) {
                val = vals[j] + rhs.vals[k];
                ++j; ++k;
            } 
            else if (col_j < col_k) {
                val = vals[j];
                ++j;
            } 
            else {
                val = rhs.vals[k];
                ++k;
            }
            
            if (std::abs(val) > 1e-10) {
                result.vals.push_back(val);
                result.col_ind.push_back(std::min(col_j, col_k));
            }
        }

        result.row_ptr[i + 1] = result.vals.size();
    }
    
    return result;
}

sparse_matrix& sparse_matrix::operator+=(const sparse_matrix& rhs) {
    *this = *this + rhs;
    return *this;
}

sparse_matrix sparse_matrix::operator-(const sparse_matrix& rhs) const {
    if (rows != rhs.rows || cols != rhs.cols) {
        throw std::runtime_error("sparse_matrix operator-: dimension mismatch");
    }
    
    sparse_matrix result(rows, cols);
    result.row_ptr[0] = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        std::size_t j = row_ptr[i], k = rhs.row_ptr[i];
        while (j < row_ptr[i + 1] || k < rhs.row_ptr[i + 1]) {
            std::size_t col_j = (j < row_ptr[i + 1]) ? col_ind[j] : cols;
            std::size_t col_k = (k < rhs.row_ptr[i + 1]) ? rhs.col_ind[k] : cols;
            double val = 0.0;
            if (col_j == col_k) {
                val = vals[j] - rhs.vals[k];
                ++j; ++k;
            } 
            else if (col_j < col_k) {
                val = vals[j];
                ++j;
            } 
            else {
                val = -rhs.vals[k];
                ++k;
            }
            
            if (std::abs(val) > 1e-10) {
                result.vals.push_back(val);
                result.col_ind.push_back(std::min(col_j, col_k));
            }
        }

        result.row_ptr[i + 1] = result.vals.size();
    }

    return result;
}

sparse_matrix& sparse_matrix::operator-=(const sparse_matrix& rhs) {
    *this = *this - rhs;
    return *this;
}

vector matrix_vector_multiply(const dense_matrix& matrix, const vector& vec) {
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        throw std::runtime_error("matrix_vector_multiply: empty matrix");
    }
    if (matrix.cols() != vec.size()) {
        throw std::runtime_error("matrix_vector_multiply: dimension mismatch");
    }
    
    vector ans(matrix.rows(), 0.0);
    auto compute = [&matrix, &vec, &ans](double& y) {
        std::size_t i = &y - ans.data();
        for (std::size_t j = 0; j < matrix.cols(); ++j) {
            y += matrix(i, j) * vec[j];
        }
    
        if (!std::isfinite(y)) {
            throw std::runtime_error("matrix_vector_multiply: non-finite result");
        }
    };
    
    if (matrix.rows() > 1000) {
        std::for_each(std::execution::par_unseq, ans.begin(), ans.end(), compute);
    } 
    else {
        std::for_each(ans.begin(), ans.end(), compute);
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
    
    dense_matrix C(A.rows(), B.cols());
    auto compute = [&A, &B, &C](std::vector<double>::iterator row) {
        std::size_t i = (row - C.data().begin()) / C.cols();
        for (std::size_t j = 0; j < B.cols(); ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
    
            if (!std::isfinite(sum)) {
                throw std::runtime_error("matrix_multiply: non-finite result");
            }
    
            (*row) = sum;
            ++row;
        }
    };
    
    if (A.rows() * B.cols() > 1000000) {
        std::for_each(std::execution::par_unseq, C.data().begin(), C.data().end(), compute);
    } 
    else {
        std::for_each(C.data().begin(), C.data().end(), compute);
    }
    
    return C;
}

sparse_matrix sparse_matrix_multiply(const sparse_matrix& A, const sparse_matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("sparse_matrix_multiply: dimension mismatch");
    }
    
    sparse_matrix C(A.rows, B.cols);
    // Symbolic phase: compute non-zero structure
    std::vector<std::size_t> nnz_per_row(A.rows, 0);
    std::vector<bool> seen(B.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        std::fill(seen.begin(), seen.end(), false);
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            std::size_t k = A.col_ind[j];
            for (std::size_t m = B.row_ptr[k]; m < B.row_ptr[k + 1]; ++m) {
                if (!seen[B.col_ind[m]]) {
                    seen[B.col_ind[m]] = true;
                    ++nnz_per_row[i];
                }
            }
        }
    }
    
    C.row_ptr[0] = 0;
    for (std::size_t i = 0; i < A.rows; ++i) {
        C.row_ptr[i + 1] = C.row_ptr[i] + nnz_per_row[i];
    }
    
    C.vals.resize(C.row_ptr[A.rows]);
    C.col_ind.resize(C.row_ptr[A.rows]);
    // Numeric phase: compute values
    std::vector<std::size_t> pos = C.row_ptr;
    for (std::size_t i = 0; i < A.rows; ++i) {
        std::vector<std::pair<std::size_t, double>> row(B.cols, {0, 0.0});
        std::size_t row_nnz = 0;
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            std::size_t k = A.col_ind[j];
            double val = A.vals[j];
            for (std::size_t m = B.row_ptr[k]; m < B.row_ptr[k + 1]; ++m) {
                std::size_t col = B.col_ind[m];
                if (row[col].first == 0) {
                    row[col].first = 1;
                    row[row_nnz++].first = col;
                }
    
                row[col].second += val * B.vals[m];
            }
        }
    
        std::sort(row.begin(), row.begin() + row_nnz);
        for (std::size_t j = 0; j < row_nnz; ++j) {
            if (std::abs(row[j].second) > 1e-10) {
                C.vals[pos[i]] = row[j].second;
                C.col_ind[pos[i]] = row[j].first;
                ++pos[i];
            }
        }
    
        C.row_ptr[i + 1] = pos[i];
    }
    
    C.vals.resize(C.row_ptr[A.rows]);
    C.col_ind.resize(C.row_ptr[A.rows]);
    C.validate();
    
    return C;
}

dense_matrix transpose(const dense_matrix& A) {
    if (A.rows() == 0 || A.cols() == 0) {
        throw std::runtime_error("transpose: empty matrix");
    }
    
    dense_matrix B(A.cols(), A.rows());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            B(j, i) = A(i, j);
        }
    }
    
    return B;
}

sparse_matrix sparse_transpose(const sparse_matrix& A) {
    if (A.rows == 0 || A.cols == 0) {
        throw std::runtime_error("sparse_transpose: empty matrix");
    }
    
    sparse_matrix B(A.cols, A.rows);
    std::vector<std::size_t> count(A.cols + 1, 0);
    for (std::size_t k = 0; k < A.col_ind.size(); ++k) {
        ++count[A.col_ind[k] + 1];
    }
    
    std::partial_sum(count.begin(), count.end(), count.begin());
    B.vals.resize(A.vals.size());
    B.col_ind.resize(A.col_ind.size());
    B.row_ptr = count;
    std::vector<std::size_t> indices = count;
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            std::size_t j = A.col_ind[k];
            std::size_t idx = indices[j]++;
            B.col_ind[idx] = i;
            B.vals[idx] = A.vals[k];
        }
    }
    
    B.validate();
    return B;
}

dense_matrix operator+(const dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("operator+: dimension mismatch");
    }
    
    dense_matrix C(A.rows(), A.cols());
    auto compute = [&A, &B, &C](double& c) {
        std::size_t idx = &c - C.data().data();
        c = A.data()[idx] + B.data()[idx];
        if (!std::isfinite(c)) {
            throw std::runtime_error("operator+: non-finite result");
        }
    };
    
    if (A.rows() * A.cols() > 1000000) {
        std::for_each(std::execution::par_unseq, C.data().begin(), C.data().end(), compute);
    } 
    else {
        std::for_each(C.data().begin(), C.data().end(), compute);
    }
    
    return C;
}

dense_matrix& operator+=(dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("operator+=: dimension mismatch");
    }
    
    auto compute = [&A, &B](double& a) {
        std::size_t idx = &a - A.data().data();
        a += B.data()[idx];
        if (!std::isfinite(a)) {
            throw std::runtime_error("operator+=: non-finite result");
        }
    };
    
    if (A.rows() * A.cols() > 1000000) {
        std::for_each(std::execution::par_unseq, A.data().begin(), A.data().end(), compute);
    } 
    else {
        std::for_each(A.data().begin(), A.data().end(), compute);
    }
    
    return A;
}

dense_matrix operator-(const dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("operator-: dimension mismatch");
    }
    
    dense_matrix C(A.rows(), A.cols());
    auto compute = [&A, &B, &C](double& c) {
        std::size_t idx = &c - C.data().data();
        c = A.data()[idx] - B.data()[idx];
        if (!std::isfinite(c)) {
            throw std::runtime_error("operator-: non-finite result");
        }
    };
    
    if (A.rows() * A.cols() > 1000000) {
        std::for_each(std::execution::par_unseq, C.data().begin(), C.data().end(), compute);
    } 
    else {
        std::for_each(C.data().begin(), C.data().end(), compute);
    }
    
    return C;
}

dense_matrix& operator-=(dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("operator-=: dimension mismatch");
    }
    
    auto compute = [&A, &B](double& a) {
        std::size_t idx = &a - A.data().data();
        a -= B.data()[idx];
        if (!std::isfinite(a)) {
            throw std::runtime_error("operator-=: non-finite result");
        }
    };
    
    if (A.rows() * A.cols() > 1000000) {
        std::for_each(std::execution::par_unseq, A.data().begin(), A.data().end(), compute);
    } 
    else {
        std::for_each(A.data().begin(), A.data().end(), compute);
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

sparse_matrix Identity(std::size_t n) {
    if (n == 0) {
        throw std::runtime_error("Identity: dimension must be non-zero");
    }
    
    sparse_matrix identity(n, n);
    identity.vals.resize(n, 1.0);
    identity.col_ind.resize(n);
    identity.row_ptr.resize(n + 1);
    for (std::size_t i = 0; i < n; ++i) {
        identity.col_ind[i] = i;
        identity.row_ptr[i] = i;
    }
    
    identity.row_ptr[n] = n;
    return identity;
}

sparse_matrix build_adj_matrix(std::size_t num_vertices, const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    if (!validate(edges, num_vertices)) {
        throw std::runtime_error("build_adj_matrix: invalid edges");
    }
    
    sparse_matrix adj(num_vertices, num_vertices);
    std::vector<std::vector<std::size_t>> adj_list(num_vertices);
    for (const auto& [u, v] : edges) {
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    
    adj.row_ptr[0] = 0;
    for (std::size_t i = 0; i < num_vertices; ++i) {
        std::sort(adj_list[i].begin(), adj_list[i].end());
        for (std::size_t j : adj_list[i]) {
            adj.vals.push_back(1.0);
            adj.col_ind.push_back(j);
        }
        adj.row_ptr[i + 1] = adj.vals.size();
    }
    
    adj.validate();
    return adj;
}

vector compute_degrees(const sparse_matrix& A) {
    vector degrees(A.rows, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) {
        degrees[i] = static_cast<double>(A.row_ptr[i + 1] - A.row_ptr[i]);
    }
    
    return degrees;
}

sparse_matrix laplacian_matrix(const sparse_matrix& A) {
    vector degrees = compute_degrees(A);
    sparse_matrix L(A.rows, A.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        L.vals.push_back(degrees[i]);
        L.col_ind.push_back(i);
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            L.vals.push_back(-A.vals[j]);
            L.col_ind.push_back(A.col_ind[j]);
        }
    
        L.row_ptr[i + 1] = L.vals.size();
    }
    
    L.validate();
    return L;
}

sparse_matrix normalized_laplacian_matrix(const sparse_matrix& A) {
    vector degrees = compute_degrees(A);
    sparse_matrix L = laplacian_matrix(A);
    sparse_matrix norm_L(A.rows, A.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        double sqrt_deg_i = std::sqrt(degrees[i]);
        if (sqrt_deg_i == 0.0) {
            continue;
        }
    
        for (std::size_t j = L.row_ptr[i]; j < L.row_ptr[i + 1]; ++j) {
            double sqrt_deg_j = std::sqrt(degrees[L.col_ind[j]]);
            if (sqrt_deg_j == 0.0) {
                continue;
            }
    
            double val = L.vals[j] / (sqrt_deg_i * sqrt_deg_j);
            if (std::abs(val) > 1e-10) {
                norm_L.vals.push_back(val);
                norm_L.col_ind.push_back(L.col_ind[j]);
            }
        }
    
        norm_L.row_ptr[i + 1] = norm_L.vals.size();
    }
    
    norm_L.validate();
    return norm_L;
}

bool validate(const std::vector<std::pair<std::size_t, std::size_t>>& edges, std::size_t num_vertices) {
    for (const auto& [u, v] : edges) {
        if (u >= num_vertices || v >= num_vertices) {
            return false;
        }
    }
    
    return true;
}

dense_matrix elementwise_multiply(const dense_matrix& A, const dense_matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("elementwise_multiply: dimension mismatch");
    }
    
    dense_matrix C(A.rows(), A.cols());
    auto compute = [&A, &B, &C](double& c) {
        std::size_t idx = &c - C.data().data();
        c = A.data()[idx] * B.data()[idx];
        if (!std::isfinite(c)) {
            throw std::runtime_error("elementwise_multiply: non-finite result");
        }
    };
    
    if (A.rows() * A.cols() > 1000000) {
        std::for_each(std::execution::par_unseq, C.data().begin(), C.data().end(), compute);
    } 
    else {
        std::for_each(C.data().begin(), C.data().end(), compute);
    }
    
    return C;
}

double frobenius_norm(const dense_matrix& A) {
    if (A.rows() == 0 || A.cols() == 0) {
        throw std::runtime_error("frobenius_norm: empty matrix");
    }
    
    double sum = 0.0;
    auto compute = [&sum](double x) {
        sum += x * x;
        if (!std::isfinite(sum)) {
            throw std::runtime_error("frobenius_norm: non-finite result");
        }
    };
    
    if (A.data().size() > 1000000) {
        std::for_each(std::execution::par_unseq, A.data().begin(), A.data().end(), compute);
    } 
    else {
        std::for_each(A.data().begin(), A.data().end(), compute);
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

dense_matrix to_dense(const sparse_matrix& A) {
    dense_matrix dense(A.rows, A.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            dense(i, A.col_ind[j]) = A.vals[j];
        }
    }
    
    return dense;
}

bool is_valid(const dense_matrix& A) {
    if (A.rows() == 0 || A.cols() == 0) {
        return true;
    }
    
    for (double x : A.data()) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    
    return true;
}

bool is_symmetric(const sparse_matrix& A) {
    if (A.rows != A.cols) {
        return false;
    }
    
    sparse_matrix AT = sparse_transpose(A);
    if (AT.vals.size() != A.vals.size()) {
        return false;
    }
    
    for (std::size_t i = 0; i < A.rows; ++i) {
        if (AT.row_ptr[i] != A.row_ptr[i]) {
            return false;
        }
    
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (AT.col_ind[j] != A.col_ind[j] || std::abs(AT.vals[j] - A.vals[j]) > 1e-10) {
                return false;
            }
        }
    }
    
    return true;
}

} // namespace matrix
} // namespace gnnmath