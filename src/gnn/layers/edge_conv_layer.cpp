#include <gnnmath/gnn/layers/edge_conv_layer.hpp>
#include <gnnmath/core/random.hpp>
#include <gnnmath/math/vector.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gnnmath {
namespace gnn {

using gnnmath::feature_vec;

namespace {

/// Forward intermediates needed by edge_conv_layer::backward.
struct edge_conv_cache : layer_cache {
    std::vector<feature_vec> edge_pre;  ///< (f_i - f_j) * W per CSR entry
};

} // namespace

edge_conv_layer::edge_conv_layer(std::size_t in_dim, std::size_t out_dim, activation_type activation)
    : in_dim_(in_dim), out_dim_(out_dim), weights_(in_dim, out_dim), bias_(out_dim), activation_(activation) {
    if (in_dim == 0 || out_dim == 0) {
        throw std::runtime_error("edge_conv_layer: dimensions must be non-zero");
    }

    // Xavier/Glorot uniform initialization
    const double limit = std::sqrt(6.0 / static_cast<double>(in_dim + out_dim));
    auto data = random::uniform_vector(in_dim * out_dim, -limit, limit);
    for (std::size_t i = 0; i < in_dim; ++i) {
        for (std::size_t j = 0; j < out_dim; ++j) {
            weights_(i, j) = data[i * out_dim + j];
        }
    }
}

std::vector<feature_vec> edge_conv_layer::forward(const std::vector<feature_vec>& features,
                                            const matrix::sparse_matrix& adj) const {
    if (features.empty() || features.size() != adj.rows || (features[0].size() != in_dim_)) {
        throw std::runtime_error("edge_conv_layer forward: dimension mismatch");
    }

    std::vector<feature_vec> out(features.size(), feature_vec(out_dim_, 0.0));
    feature_vec weighted(out_dim_, 0.0);
    for (std::size_t i = 0; i < features.size(); ++i) {
        for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
            std::size_t j = adj.col_ind[k];
            for (std::size_t m = 0; m < out_dim_; ++m) {
                double sum = 0.0;
                for (std::size_t n = 0; n < in_dim_; ++n) {
                    sum += (features[i][n] - features[j][n]) * weights_(n, m);
                }

                weighted[m] = sum;
            }

            feature_vec activated = apply_activation(weighted, activation_);
            for (std::size_t m = 0; m < out_dim_; ++m) {
                out[i][m] += activated[m];
            }
        }

        for (std::size_t m = 0; m < out_dim_; ++m) {
            out[i][m] += bias_[m];
            if (!std::isfinite(out[i][m])) {
                throw std::runtime_error("edge_conv_layer forward: non-finite result");
            }
        }
    }

    return out;
}

std::vector<feature_vec> edge_conv_layer::forward_cached(const std::vector<feature_vec>& features,
                                                   const matrix::sparse_matrix& adj,
                                                   std::unique_ptr<layer_cache>& cache) const {
    if (features.empty() || features.size() != adj.rows || (features[0].size() != in_dim_)) {
        throw std::runtime_error("edge_conv_layer forward_cached: dimension mismatch");
    }

    const std::size_t num_nodes = features.size();
    auto c = std::make_unique<edge_conv_cache>();
    c->edge_pre.assign(adj.col_ind.size(), feature_vec(out_dim_, 0.0));

    // out_i = bias + sum_j activation((f_i - f_j) * W)
    std::vector<feature_vec> out(num_nodes);
    for (std::size_t i = 0; i < num_nodes; ++i) {
        out[i] = bias_;
        for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
            std::size_t j = adj.col_ind[k];
            feature_vec& pre = c->edge_pre[k];
            for (std::size_t m = 0; m < out_dim_; ++m) {
                double sum = 0.0;
                for (std::size_t n = 0; n < in_dim_; ++n) {
                    sum += (features[i][n] - features[j][n]) * weights_(n, m);
                }

                pre[m] = sum;
            }

            feature_vec activated = apply_activation(pre, activation_);
            for (std::size_t m = 0; m < out_dim_; ++m) {
                out[i][m] += activated[m];
            }
        }

        for (double x : out[i]) {
            if (!std::isfinite(x)) {
                throw std::runtime_error("edge_conv_layer forward_cached: non-finite result");
            }
        }
    }

    cache = std::move(c);
    return out;
}

std::vector<feature_vec> edge_conv_layer::backward(const std::vector<feature_vec>& delta_out,
                                             const std::vector<feature_vec>& input,
                                             const matrix::sparse_matrix& adj,
                                             const layer_cache& cache,
                                             matrix::dense_matrix& weight_grad,
                                             feature_vec& bias_grad,
                                             bool compute_delta_prev) const {
    const auto& c = dynamic_cast<const edge_conv_cache&>(cache);
    const std::size_t num_nodes = delta_out.size();

    std::vector<feature_vec> delta_prev;
    if (compute_delta_prev) {
        delta_prev.assign(num_nodes, feature_vec(in_dim_, 0.0));
    }

    feature_vec delta_edge(out_dim_);
    for (std::size_t i = 0; i < num_nodes; ++i) {
        // Bias is added outside the per-edge activation
        for (std::size_t m = 0; m < out_dim_; ++m) {
            bias_grad[m] += delta_out[i][m];
        }

        for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
            std::size_t j = adj.col_ind[k];
            // Through the per-edge activation
            for (std::size_t m = 0; m < out_dim_; ++m) {
                delta_edge[m] = delta_out[i][m] * activation_derivative(c.edge_pre[k][m], activation_);
            }

            for (std::size_t n = 0; n < in_dim_; ++n) {
                double diff = input[i][n] - input[j][n];
                for (std::size_t m = 0; m < out_dim_; ++m) {
                    weight_grad(n, m) += diff * delta_edge[m];
                }
            }

            // d(f_i - f_j)/d_f_i = 1, d(f_i - f_j)/d_f_j = -1
            if (compute_delta_prev) {
                for (std::size_t n = 0; n < in_dim_; ++n) {
                    double back = 0.0;
                    for (std::size_t m = 0; m < out_dim_; ++m) {
                        back += delta_edge[m] * weights_(n, m);
                    }

                    delta_prev[i][n] += back;
                    delta_prev[j][n] -= back;
                }
            }
        }
    }

    return delta_prev;
}

} // namespace gnn
} // namespace gnnmath
