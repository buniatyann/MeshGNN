#include <gnnmath/math/sparse_matrix.hpp>
#include <gnnmath/math/dense_matrix.hpp>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace gnnmath {
namespace matrix {

sparse_matrix::sparse_matrix(std::size_t rows, std::size_t cols) : rows(rows), cols(cols) {
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("sparse_matrix: dimensions must be non-zero");
    }

    row_ptr.resize(rows + 1, 0);
}

sparse_matrix::sparse_matrix(const dense_matrix& rhs) {
    rows = rhs.rows();
    cols = rhs.cols();
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("sparse_matrix: empty input matrix");
    }

    row_ptr.resize(rows + 1, 0);
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

    validate();
}

sparse_matrix::sparse_matrix(std::size_t rows, std::size_t cols, std::vector<double>&& values,
                             std::vector<std::size_t>&& col_indices, std::vector<std::size_t>&& row_ptrs)
    : vals(std::move(values)), col_ind(std::move(col_indices)), row_ptr(std::move(row_ptrs)), rows(rows), cols(cols) {
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
        if (row_ptr[i] < row_ptr[i - 1]) {
            throw std::runtime_error("sparse_matrix: row_ptr not non-decreasing");
        }
    }

    // Column indices must be strictly increasing within each row; the
    // merge-based operators (+, -) and is_symmetric rely on this ordering
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = row_ptr[i] + 1; j < row_ptr[i + 1]; ++j) {
            if (col_ind[j] <= col_ind[j - 1]) {
                throw std::runtime_error("sparse_matrix: columns not sorted within row");
            }
        }
    }
}

vector sparse_matrix::multiply(const vector& x) const {
    if (cols != x.size()) {
        throw std::runtime_error("sparse_matrix multiply: dimension mismatch");
    }
    
    vector ans(rows, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            ans[i] += vals[j] * x[col_ind[j]];
        }

        if (!std::isfinite(ans[i])) {
            throw std::runtime_error("sparse_matrix multiply: non-finite result");
        }
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
                ++j; 
                ++k;
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
                std::size_t actual_col = (col_j <= col_k) ? col_j : col_k;
                result.col_ind.push_back(actual_col);
            }
        }

        result.row_ptr[i + 1] = result.vals.size();
    }

    result.validate();
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
                ++j; 
                ++k;
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
                std::size_t actual_col = (col_j <= col_k) ? col_j : col_k;
                result.col_ind.push_back(actual_col);
            }
        }

        result.row_ptr[i + 1] = result.vals.size();
    }

    result.validate();
    return result;
}

sparse_matrix& sparse_matrix::operator-=(const sparse_matrix& rhs) {
    *this = *this - rhs;
    return *this;
}

sparse_matrix sparse_matrix_multiply(const sparse_matrix& A, const sparse_matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("sparse_matrix_multiply: dimension mismatch");
    }

    sparse_matrix C(A.rows, B.cols);
    C.row_ptr[0] = 0;
    // Dense workspace SpGEMM: accumulate each row into a scratch array and
    // track touched columns, so per-row work is linear in the flops instead of
    // quadratic in the row's nonzeros.
    std::vector<double> accum(B.cols, 0.0);
    std::vector<char> marked(B.cols, 0);
    std::vector<std::size_t> touched;
    for (std::size_t i = 0; i < A.rows; ++i) {
        touched.clear();
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            std::size_t k = A.col_ind[j];
            double val_a = A.vals[j];
            for (std::size_t m = B.row_ptr[k]; m < B.row_ptr[k + 1]; ++m) {
                std::size_t col = B.col_ind[m];
                if (!marked[col]) {
                    marked[col] = 1;
                    touched.push_back(col);
                }

                accum[col] += val_a * B.vals[m];
            }
        }

        std::sort(touched.begin(), touched.end());
        for (std::size_t col : touched) {
            if (std::abs(accum[col]) > 1e-10) {
                C.vals.push_back(accum[col]);
                C.col_ind.push_back(col);
            }

            accum[col] = 0.0;
            marked[col] = 0;
        }

        C.row_ptr[i + 1] = C.vals.size();
    }

    C.validate();
    return C;
}

sparse_matrix sparse_transpose(const sparse_matrix& A) {
    if (A.rows == 0 || A.cols == 0) {
        throw std::runtime_error("sparse_transpose: empty matrix");
    }

    sparse_matrix B(A.cols, A.rows);
    std::vector<std::size_t> count(A.cols, 0);
    for (std::size_t k = 0; k < A.col_ind.size(); ++k) {
        ++count[A.col_ind[k]];
    }

    B.row_ptr.resize(A.cols + 1);
    B.row_ptr[0] = 0;
    for (std::size_t i = 1; i <= A.cols; ++i) {
        B.row_ptr[i] = B.row_ptr[i - 1] + count[i - 1];
    }

    B.vals.resize(A.vals.size());
    B.col_ind.resize(A.col_ind.size());
    std::vector<std::size_t> indices = B.row_ptr;
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
    identity.validate();

    return identity;
}

sparse_matrix build_adj_matrix(std::size_t num_vertices, const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    if (num_vertices == 0) {
        throw std::runtime_error("build_adj_matrix: zero vertices");
    }

    for (const auto& [u, v] : edges) {
        if (u >= num_vertices || v >= num_vertices) {
            throw std::runtime_error("build_adj_matrix: invalid vertex index");
        }
    }

    sparse_matrix adj(num_vertices, num_vertices);
    std::vector<std::vector<std::size_t>> adj_list(num_vertices);
    for (const auto& [u, v] : edges) {
        adj_list[u].push_back(v);
        if (u != v) {
            adj_list[v].push_back(u); // Ensure symmetry
        }
    }

    adj.row_ptr[0] = 0;
    for (std::size_t i = 0; i < num_vertices; ++i) {
        std::sort(adj_list[i].begin(), adj_list[i].end());
        auto last = std::unique(adj_list[i].begin(), adj_list[i].end()); // Remove duplicates
        adj_list[i].erase(last, adj_list[i].end());
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
    if (A.rows == 0) {
        throw std::runtime_error("compute_degrees: empty matrix");
    }

    vector degrees(A.rows, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) {
        // Weighted degree: sum of the row's values (equals the neighbor count
        // for a 0/1 adjacency matrix)
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            degrees[i] += A.vals[j];
        }
    }

    return degrees;
}

sparse_matrix laplacian_matrix(const sparse_matrix& A) {
    if (A.rows == 0 || A.cols == 0) {
        throw std::runtime_error("laplacian_matrix: empty matrix");
    }

    if (A.rows != A.cols) {
        throw std::runtime_error("laplacian_matrix: matrix must be square");
    }

    vector degrees = compute_degrees(A);
    sparse_matrix L(A.rows, A.cols);
    L.row_ptr[0] = 0;
    for (std::size_t i = 0; i < A.rows; ++i) {
        // Emit entries in sorted column order, merging the diagonal (degree)
        // into its proper position among the -A entries
        bool diag_emitted = false;
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            std::size_t col = A.col_ind[j];
            if (!diag_emitted && col >= i) {
                double diag = degrees[i] - (col == i ? A.vals[j] : 0.0);
                if (std::abs(diag) > 1e-10) {
                    L.vals.push_back(diag);
                    L.col_ind.push_back(i);
                }

                diag_emitted = true;
                if (col == i) {
                    continue;
                }
            }

            L.vals.push_back(-A.vals[j]);
            L.col_ind.push_back(col);
        }

        if (!diag_emitted && std::abs(degrees[i]) > 1e-10) {
            L.vals.push_back(degrees[i]);
            L.col_ind.push_back(i);
        }

        L.row_ptr[i + 1] = L.vals.size();
    }

    L.validate();
    return L;
}

sparse_matrix normalized_laplacian_matrix(const sparse_matrix& A) {
    if (A.rows == 0 || A.cols == 0) {
        throw std::runtime_error("normalized_laplacian_matrix: empty matrix");
    }
    if (A.rows != A.cols) {
        throw std::runtime_error("normalized_laplacian_matrix: matrix must be square");
    }

    vector degrees = compute_degrees(A);
    sparse_matrix L = laplacian_matrix(A);
    sparse_matrix norm_L(A.rows, A.cols);
    norm_L.row_ptr[0] = 0;
    for (std::size_t i = 0; i < A.rows; ++i) {
        double sqrt_deg_i = degrees[i] > 0 ? std::sqrt(degrees[i]) : 0.0;
        if (sqrt_deg_i == 0.0) {
            norm_L.row_ptr[i + 1] = norm_L.vals.size();
            continue;
        }

        for (std::size_t j = L.row_ptr[i]; j < L.row_ptr[i + 1]; ++j) {
            double sqrt_deg_j = degrees[L.col_ind[j]] > 0 ? std::sqrt(degrees[L.col_ind[j]]) : 0.0;
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

sparse_matrix normalized_adjacency(const sparse_matrix& A) {
    if (A.rows == 0 || A.cols == 0) {
        throw std::runtime_error("normalized_adjacency: empty matrix");
    }
    if (A.rows != A.cols) {
        throw std::runtime_error("normalized_adjacency: matrix must be square");
    }

    // A + I (merge-based operator+ keeps sorted CSR), then scale each entry
    // (i, j) by 1 / sqrt(deg_i * deg_j) with degrees of A + I
    sparse_matrix ai = A + Identity(A.rows);
    vector degrees = compute_degrees(ai);
    for (std::size_t i = 0; i < ai.rows; ++i) {
        for (std::size_t j = ai.row_ptr[i]; j < ai.row_ptr[i + 1]; ++j) {
            double d = degrees[i] * degrees[ai.col_ind[j]];
            if (d <= 0.0) {
                throw std::runtime_error("normalized_adjacency: non-positive degree");
            }

            ai.vals[j] /= std::sqrt(d);
        }
    }

    ai.validate();
    return ai;
}

bool validate(const std::vector<std::pair<std::size_t, std::size_t>>& edges, std::size_t num_vertices) {
    for (const auto& [u, v] : edges) {
        if (u >= num_vertices || v >= num_vertices) {
            return false;
        }
    }

    return true;
}

dense_matrix to_dense(const sparse_matrix& A) {
    if (A.rows == 0 || A.cols == 0) {
        throw std::runtime_error("to_dense: empty matrix");
    }

    dense_matrix dense(A.rows, A.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            dense(i, A.col_ind[j]) = A.vals[j];
        }
    }

    return dense;
}

bool is_symmetric(const sparse_matrix& A) {
    if (A.rows == 0 || A.cols == 0) {
        throw std::runtime_error("is_symmetric: empty matrix");
    }
    if (A.rows != A.cols) {
        return false;
    }

    sparse_matrix AT = sparse_transpose(A);
    if (AT.vals.size() != A.vals.size() || AT.col_ind.size() != A.col_ind.size()) {
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
