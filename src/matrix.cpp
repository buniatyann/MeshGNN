#include "../include/gnnmath/matrix.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <execution>

namespace gnnmath {
namespace matrix {

CSR::CSR(std::size_t r, std::size_t c) : rows(r), cols(c) {
    if (r == 0 || c == 0) {
        throw std::runtime_error("CSR: dimensions must be non-zero (rows=" + std::to_string(r) +
                                 ", cols=" + std::to_string(c) + ")");
    }
    
    row_ptr.resize(r + 1, 0);
}

CSR::CSR(const dense_matrix& rhs) {
    if (rhs.empty()) {
        throw std::runtime_error("CSR: cannot convert empty dense matrix");
    }
    
    rows = rhs.size();
    cols = rhs[0].size();
    for (const auto& row : rhs) {
        if (row.size() != cols) {
            throw std::runtime_error("CSR: inconsistent row sizes in dense matrix");
        }
    }

    row_ptr.resize(rows + 1);
    row_ptr[0] = 0;

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            if (std::abs(rhs[i][j]) > 1e-10) { // avoid floating-point noise
                if (!std::isfinite(rhs[i][j])) {
                    throw std::runtime_error("CSR: NaN or infinity detected at [" +
                                             std::to_string(i) + "," + std::to_string(j) + "]");
                }
    
                vals.push_back(rhs[i][j]);
                col_ind.push_back(j);
            }
        }
    
        row_ptr[i + 1] = vals.size();
    }
}

vector CSR::multiply(const vector& x) const {
    if (cols != x.size()) {
        throw std::runtime_error("CSR multiply: matrix columns (" + std::to_string(cols) +
                                 ") must match vector size (" + std::to_string(x.size()) + ")");
    }
    vector ans(rows, 0.0);
    std::for_each(std::execution::par_unseq, ans.begin(), ans.end(),
                  [this, &x, &ans](double& y) {
                      std::size_t i = &y - ans.data();
                      for (std::size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                          y += vals[j] * x[col_ind[j]];
                      }
                      if (!std::isfinite(y)) {
                          throw std::runtime_error("CSR multiply: non-finite result at row " +
                                                   std::to_string(i));
                      }
                  });
    
    return ans;
}

vector matrix_vector_multiply(const dense_matrix& matrix, const vector& vec) {
    if (matrix.empty() || matrix[0].empty()) {
        throw std::runtime_error("matrix_vector_multiply: empty matrix");
    }
    
    if (matrix[0].size() != vec.size()) {
        throw std::runtime_error("matrix_vector_multiply: matrix columns (" +
                                 std::to_string(matrix[0].size()) + ") must match vector size (" +
                                 std::to_string(vec.size()) + ")");
    }
    
    for (const auto& row : matrix) {
        if (row.size() != matrix[0].size()) {
            throw std::runtime_error("matrix_vector_multiply: inconsistent row sizes");
        }
    }

    vector ans(matrix.size(), 0.0);
    std::for_each(std::execution::par_unseq, ans.begin(), ans.end(),
                  [&matrix, &vec, &ans](double& y) {
                      std::size_t i = &y - ans.data();
                      for (std::size_t j = 0; j < matrix[0].size(); ++j) {
                          y += matrix[i][j] * vec[j];
                      }
                      
                      if (!std::isfinite(y)) {
                          throw std::runtime_error("matrix_vector_multiply: non-finite result at row " +
                                                   std::to_string(i));
                      }
                  });
    
    return ans;
}

vector sparse_matrix_vector_multiply(const CSR& matrix, const vector& vec) {
    return matrix.multiply(vec); // Delegate to CSR::multiply
}

dense_matrix operator*(const dense_matrix& A, const dense_matrix& B) {
    if (A.empty() || A[0].empty() || B.empty() || B[0].empty()) {
        throw std::runtime_error("matrix_multiply: empty matrix");
    }

    if (A[0].size() != B.size()) {
        throw std::runtime_error("matrix_multiply: A columns (" + std::to_string(A[0].size()) +
                                 ") must match B rows (" + std::to_string(B.size()) + ")");
    }
    
    for (const auto& row : A) {
        if (row.size() != A[0].size()) {
            throw std::runtime_error("matrix_multiply: inconsistent row sizes in A");
        }
    }
    
    for (const auto& row : B) {
        if (row.size() != B[0].size()) {
            throw std::runtime_error("matrix_multiply: inconsistent row sizes in B");
        }
    }

    dense_matrix C(A.size(), std::vector<double>(B[0].size(), 0.0));
    std::for_each(std::execution::par_unseq, C.begin(), C.end(),
                  [&A, &B, &C](std::vector<double>& row) {
                      std::size_t i = &row - C.data();
                      for (std::size_t j = 0; j < B[0].size(); ++j) {
                          for (std::size_t k = 0; k < A[0].size(); ++k) {
                              row[j] += A[i][k] * B[k][j];
                          }
    
                          if (!std::isfinite(row[j])) {
                              throw std::runtime_error("matrix_multiply: non-finite result at [" +
                                                       std::to_string(i) + "," + std::to_string(j) + "]");
                          }
                      }
                  });
    
    return C;
}

CSR sparse_matrix_multiply(const CSR& A, const CSR& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("sparse_matrix_multiply: A columns (" + std::to_string(A.cols) +
                                 ") must match B rows (" + std::to_string(B.rows) + ")");
    }
    
    CSR C(A.rows, B.cols);
    std::vector<std::size_t> row_nnz(A.rows, 0);

    // Count non-zeros per row
    for (std::size_t i = 0; i < A.rows; ++i) {
        std::vector<bool> non_zero(B.cols, false);
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            std::size_t k = A.col_ind[j];
            for (std::size_t l = B.row_ptr[k]; l < B.row_ptr[k + 1]; ++l) {
                if (!non_zero[B.col_ind[l]]) {
                    non_zero[B.col_ind[l]] = true;
                    ++row_nnz[i];
                }
            }
        }
    }

    // Allocate C
    C.row_ptr[0] = 0;
    for (std::size_t i = 0; i < A.rows; ++i) {
        C.row_ptr[i + 1] = C.row_ptr[i] + row_nnz[i];
    }
    
    C.vals.resize(C.row_ptr[A.rows]);
    C.col_ind.resize(C.row_ptr[A.rows]);

    // Compute non-zero elements
    for (std::size_t i = 0; i < A.rows; ++i) {
        std::vector<double> row(B.cols, 0.0);
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            std::size_t k = A.col_ind[j];
            for (std::size_t l = B.row_ptr[k]; l < B.row_ptr[k + 1]; ++l) {
                row[B.col_ind[l]] += A.vals[j] * B.vals[l];
            }
        }
    
        for (std::size_t j = 0; j < B.cols; ++j) {
            if (std::abs(row[j]) > 1e-10) {
                C.vals[C.row_ptr[i] + row_nnz[i]] = row[j];
                C.col_ind[C.row_ptr[i] + row_nnz[i]] = j;
                ++row_nnz[i];
            }
        }
    }
    
    return C;
}

dense_matrix transpose(const dense_matrix& A) {
    if (A.empty() || A[0].empty()) {
        throw std::runtime_error("transpose: empty matrix");
    }
    
    std::size_t rows = A.size(), cols = A[0].size();
    for (const auto& row : A) {
        if (row.size() != cols) {
            throw std::runtime_error("transpose: inconsistent row sizes");
        }
    }

    dense_matrix B(cols, std::vector<double>(rows, 0.0));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            B[j][i] = A[i][j];
        }
    }
    
    return B;
}

CSR sparse_transpose(const CSR& A) {
    CSR B(A.cols, A.rows);
    std::vector<std::vector<std::pair<std::size_t, double>>> temp(A.cols);
    
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            temp[A.col_ind[j]].emplace_back(i, A.vals[j]);
        }
    }

    B.row_ptr[0] = 0;
    std::size_t pos = 0;
    for (std::size_t i = 0; i < A.cols; ++i) {
        for (const auto& [col, val] : temp[i]) {
            B.col_ind.push_back(col);
            B.vals.push_back(val);
            ++pos;
        }
        B.row_ptr[i + 1] = pos;
    }
    
    return B;
}

dense_matrix operator+(const dense_matrix& A, const dense_matrix& B) {
    if (A.size() != B.size() || A.empty() || A[0].size() != B[0].size()) {
        throw std::runtime_error("operator+: dimension mismatch (A: " +
                                 std::to_string(A.size()) + "x" + std::to_string(A[0].size()) +
                                 ", B: " + std::to_string(B.size()) + "x" +
                                 std::to_string(B[0].size()) + ")");
    }
    
    dense_matrix C(A.size(), std::vector<double>(A[0].size()));
    std::for_each(std::execution::par_unseq, C.begin(), C.end(),
                  [&A, &B, &C](std::vector<double>& row) {
                      std::size_t i = &row - C.data();
                      std::transform(A[i].begin(), A[i].end(), B[i].begin(), row.begin(),
                                     [](double a, double b) {
                                         double sum = a + b;
                                         if (!std::isfinite(sum)) {
                                             throw std::runtime_error("operator+: non-finite result");
                                         }
    
                                         return sum;
                                     });
                  });
    
    return C;
}

dense_matrix& operator+=(dense_matrix& A, const dense_matrix& B) {
    if (A.size() != B.size() || A.empty() || A[0].size() != B[0].size()) {
        throw std::runtime_error("operator+=: dimension mismatch");
    }
    
    std::for_each(std::execution::par_unseq, A.begin(), A.end(),
                  [&B, &A](std::vector<double>& row) {
                      std::size_t i = &row - A.data();
                      std::transform(row.begin(), row.end(), B[i].begin(), row.begin(),
                                     [](double a, double b) {
                                         double sum = a + b;
                                         if (!std::isfinite(sum)) {
                                             throw std::runtime_error("operator+=: non-finite result");
                                         }
                                         
                                         return sum;
                                     });
                  });
    
    return A;
}

dense_matrix operator-(const dense_matrix& A, const dense_matrix& B) {
    if (A.size() != B.size() || A.empty() || A[0].size() != B[0].size()) {
        throw std::runtime_error("operator-: dimension mismatch");
    }
    
    dense_matrix C(A.size(), std::vector<double>(A[0].size()));
    std::for_each(std::execution::par_unseq, C.begin(), C.end(),
                  [&A, &B, &C](std::vector<double>& row) {
                      std::size_t i = &row - C.data();
                      std::transform(A[i].begin(), A[i].end(), B[i].begin(), row.begin(),
                                     [](double a, double b) {
                                         double diff = a - b;
                                         if (!std::isfinite(diff)) {
                                             throw std::runtime_error("operator-: non-finite result");
                                         }
                            
                                         return diff;
                                     });
                  });
    
    return C;
}

dense_matrix& operator-=(dense_matrix& A, const dense_matrix& B) {
    if (A.size() != B.size() || A.empty() || A[0].size() != B[0].size()) {
        throw std::runtime_error("operator-=: dimension mismatch");
    }
   
    std::for_each(std::execution::par_unseq, A.begin(), A.end(),
                  [&A, &B](std::vector<double>& row) {
                      std::size_t i = &row - A.data();
                      std::transform(row.begin(), row.end(), B[i].begin(), row.begin(),
                                     [](double a, double b) {
                                         double diff = a - b;
                                         if (!std::isfinite(diff)) {
                                             throw std::runtime_error("operator-=: non-finite result");
                                         }
                                         
                                         return diff;
                                     });
                  });

    return A;
}

CSR operator+(const CSR& A, const CSR& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::runtime_error("CSR operator+: dimension mismatch");
    }
    
    CSR C(A.rows, A.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        std::vector<std::pair<std::size_t, double>> non_zeros;
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            non_zeros.emplace_back(A.col_ind[j], A.vals[j]);
        }
    
        for (std::size_t j = B.row_ptr[i]; j < B.row_ptr[i + 1]; ++j) {
            non_zeros.emplace_back(B.col_ind[j], B.vals[j]);
        }
    
        std::sort(non_zeros.begin(), non_zeros.end());
        std::size_t k = 0;
        while (k < non_zeros.size()) {
            double sum = non_zeros[k].second;
            std::size_t col = non_zeros[k].first;
            while (k + 1 < non_zeros.size() && non_zeros[k + 1].first == col) {
                sum += non_zeros[++k].second;
            }
    
            if (std::abs(sum) > 1e-10) {
                C.vals.push_back(sum);
                C.col_ind.push_back(col);
            }
    
            ++k;
        }
    
        C.row_ptr[i + 1] = C.vals.size();
    }
    
    return C;
}

CSR& operator+=(CSR& A, const CSR& B) {
    A = A + B;
    return A;
}

CSR operator-(const CSR& A, const CSR& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::runtime_error("CSR operator-: dimension mismatch");
    }
    
    CSR C(A.rows, A.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        std::vector<std::pair<std::size_t, double>> non_zeros;
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            non_zeros.emplace_back(A.col_ind[j], A.vals[j]);
        }
    
        for (std::size_t j = B.row_ptr[i]; j < B.row_ptr[i + 1]; ++j) {
            non_zeros.emplace_back(B.col_ind[j], -B.vals[j]);
        }
    
        std::sort(non_zeros.begin(), non_zeros.end());
        std::size_t k = 0;
        while (k < non_zeros.size()) {
            double sum = non_zeros[k].second;
            std::size_t col = non_zeros[k].first;
            while (k + 1 < non_zeros.size() && non_zeros[k + 1].first == col) {
                sum += non_zeros[++k].second;
            }
    
            if (std::abs(sum) > 1e-10) {
                C.vals.push_back(sum);
                C.col_ind.push_back(col);
            }
    
            ++k;
        }
    
        C.row_ptr[i + 1] = C.vals.size();
    }
    
    return C;
}

CSR& operator-=(CSR& A, const CSR& B) {
    A = A - B;
    return A;
}

dense_matrix I(std::size_t n) {
    if (n == 0) {
        throw std::runtime_error("I: dimension must be non-zero");
    }
    
    dense_matrix identity(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        identity[i][i] = 1.0;
    }
    
    return identity;
}

CSR Identity(std::size_t n) {
    if (n == 0) {
        throw std::runtime_error("Identity: dimension must be non-zero");
    }
    
    CSR identity(n, n);
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

CSR build_adj_matrix(std::size_t num_vertices, const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    if (!validate(edges, num_vertices)) {
        throw std::runtime_error("build_adj_matrix: invalid edges");
    }
    
    CSR adj(num_vertices, num_vertices);
    std::vector<std::vector<std::size_t>> adj_list(num_vertices);
    for (const auto& [u, v] : edges) {
        adj_list[u].push_back(v);
        adj_list[v].push_back(u); // Undirected graph
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
    
    return adj;
}

vector compute_degrees(const CSR& A) {
    vector degrees(A.rows, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) {
        degrees[i] = static_cast<double>(A.row_ptr[i + 1] - A.row_ptr[i]);
    }
    
    return degrees;
}

CSR laplacian_matrix(const CSR& A) {
    vector degrees = compute_degrees(A);
    CSR L(A.rows, A.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        L.vals.push_back(degrees[i]);
        L.col_ind.push_back(i);
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            L.vals.push_back(-A.vals[j]);
            L.col_ind.push_back(A.col_ind[j]);
        }
    
        L.row_ptr[i + 1] = L.vals.size();
    }
    
    return L;
}

CSR normalized_laplacian_matrix(const CSR& A) {
    vector degrees = compute_degrees(A);
    CSR L = laplacian_matrix(A);
    CSR norm_L(A.rows, A.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        double sqrt_deg_i = std::sqrt(degrees[i]);
        if (sqrt_deg_i == 0.0) {
            continue; // Skip isolated vertices
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
    if (A.size() != B.size() || A.empty() || A[0].size() != B[0].size()) {
        throw std::runtime_error("elementwise_multiply: dimension mismatch");
    }
    
    dense_matrix C(A.size(), std::vector<double>(A[0].size()));
    std::for_each(std::execution::par_unseq, C.begin(), C.end(),
                  [&A, &B, &C](std::vector<double>& row) {
                      std::size_t i = &row - C.data();
                      std::transform(A[i].begin(), A[i].end(), B[i].begin(), row.begin(),
                                     [](double a, double b) {
                                         double prod = a * b;
                                         if (!std::isfinite(prod)) {
                                             throw std::runtime_error("elementwise_multiply: non-finite result");
                                         }
                                         return prod;
                                     });
                  });
    
    return C;
}

double frobenius_norm(const dense_matrix& A) {
    if (A.empty() || A[0].empty()) {
        throw std::runtime_error("frobenius_norm: empty matrix");
    }
    
    double sum = 0.0;
    for (const auto& row : A) {
        for (double x : row) {
            sum += x * x;
            if (!std::isfinite(sum)) {
                throw std::runtime_error("frobenius_norm: non-finite result");
            }
        }
    }
    
    return std::sqrt(sum);
}

vector extract_diagonal(const dense_matrix& A) {
    if (A.empty() || A[0].empty()) {
        throw std::runtime_error("extract_diagonal: empty matrix");
    }
    
    if (A.size() != A[0].size()) {
        throw std::runtime_error("extract_diagonal: matrix must be square");
    }
    
    vector diag(A.size());
    for (std::size_t i = 0; i < A.size(); ++i) {
        diag[i] = A[i][i];
    }
    
    return diag;
}

dense_matrix to_dense(const CSR& A) {
    dense_matrix dense(A.rows, std::vector<double>(A.cols, 0.0));
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            dense[i][A.col_ind[j]] = A.vals[j];
        }
    }
    
    return dense;
}

bool is_valid(const dense_matrix& A) {
    if (A.empty()) {
        return true;
    }
    
    std::size_t cols = A[0].size();
    for (const auto& row : A) {
        if (row.size() != cols) {
            return false;
        }
    
        for (double x : row) {
            if (!std::isfinite(x)) {
                return false;
            }
        }
    }
    
    return true;
}

bool is_symmetric(const CSR& A) {
    if (A.rows != A.cols) {
        return false;
    }
    
    CSR AT = sparse_transpose(A);
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

// void print_matrix(const dense_matrix& A) {
//     for (const auto& row : A) {
//         for (double x : row) {
//             std::cout << x << " ";
//         }
//         std::cout << std::endl;
//     }
// }

} // namespace matrix
} // namespace gnnmath