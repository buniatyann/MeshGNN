#include <gnnmath/gnn/layers/gcn_layer.hpp>
#include <gnnmath/core/random.hpp>
#include <gnnmath/math/vector.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gnnmath {
namespace gnn {

using gnnmath::feature_vec;

namespace {

/// Forward intermediates needed by gcn_layer::backward.
struct gcn_cache : layer_cache {
    std::vector<feature_vec> pre_act;    ///< A*H*W + b per node
    std::vector<feature_vec> aggregate;  ///< A*H per node
};

} // namespace

gcn_layer::gcn_layer(std::size_t in_dim, std::size_t out_dim, activation_type activation)
    : in_dim_(in_dim), out_dim_(out_dim), weights_(in_dim, out_dim), bias_(out_dim), activation_(activation) {
    if (in_dim == 0 || out_dim == 0) {
        throw std::runtime_error("gcn_layer: dimensions must be non-zero");
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

std::vector<feature_vec> gcn_layer::forward(const std::vector<feature_vec>& features,
                                      const matrix::sparse_matrix& adj) const {
    if (features.empty() || features.size() != adj.rows || (features[0].size() != in_dim_)) {
        throw std::runtime_error("gcn_layer forward: dimension mismatch");
    }

    const std::size_t num_nodes = features.size();
    std::vector<feature_vec> out(num_nodes, feature_vec(out_dim_, 0.0));

    // GCN: H' = σ(A * H * W + b)
    // For each node i, aggregate features from neighbors then transform
    for (std::size_t i = 0; i < num_nodes; ++i) {
        // Aggregate features from neighbors (A * H)_i = sum_{j in N(i)} h_j
        feature_vec aggregated(in_dim_, 0.0);
        for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
            std::size_t j = adj.col_ind[k];
            double weight = adj.vals[k];
            for (std::size_t f = 0; f < in_dim_; ++f) {
                aggregated[f] += weight * features[j][f];
            }
        }

        // Transform: (aggregated * W + b)
        for (std::size_t j = 0; j < out_dim_; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < in_dim_; ++k) {
                sum += aggregated[k] * weights_(k, j);
            }

            out[i][j] = sum + bias_[j];
            if (!std::isfinite(out[i][j])) {
                throw std::runtime_error("gcn_layer forward: non-finite result");
            }
        }
    }

    // Apply activation
    for (auto& row : out) {
        switch (activation_) {
            case activation_type::RELU:
                row = gnnmath::vector::relu(row);
                break;
            case activation_type::MISH:
                row = gnnmath::vector::mish(row);
                break;
            case activation_type::SIGMOID:
                row = gnnmath::vector::sigmoid(row);
                break;
            case activation_type::GELU:
                row = gnnmath::vector::gelu(row);
                break;
        }
    }

    return out;
}

std::vector<feature_vec> gcn_layer::forward_cached(const std::vector<feature_vec>& features,
                                             const matrix::sparse_matrix& adj,
                                             std::unique_ptr<layer_cache>& cache) const {
    if (features.empty() || features.size() != adj.rows || (features[0].size() != in_dim_)) {
        throw std::runtime_error("gcn_layer forward_cached: dimension mismatch");
    }

    const std::size_t num_nodes = features.size();
    auto c = std::make_unique<gcn_cache>();
    c->pre_act.assign(num_nodes, feature_vec(out_dim_, 0.0));
    c->aggregate.assign(num_nodes, feature_vec(in_dim_, 0.0));

    std::vector<feature_vec> out(num_nodes);
    for (std::size_t i = 0; i < num_nodes; ++i) {
        feature_vec& aggregated = c->aggregate[i];
        for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
            std::size_t j = adj.col_ind[k];
            double weight = adj.vals[k];
            for (std::size_t n = 0; n < in_dim_; ++n) {
                aggregated[n] += weight * features[j][n];
            }
        }

        for (std::size_t m = 0; m < out_dim_; ++m) {
            double sum = 0.0;
            for (std::size_t n = 0; n < in_dim_; ++n) {
                sum += aggregated[n] * weights_(n, m);
            }

            c->pre_act[i][m] = sum + bias_[m];
            if (!std::isfinite(c->pre_act[i][m])) {
                throw std::runtime_error("gcn_layer forward_cached: non-finite result");
            }
        }

        out[i] = apply_activation(c->pre_act[i], activation_);
    }

    cache = std::move(c);
    return out;
}

std::vector<feature_vec> gcn_layer::backward(const std::vector<feature_vec>& delta_out,
                                       const std::vector<feature_vec>& /*input*/,
                                       const matrix::sparse_matrix& adj,
                                       const layer_cache& cache,
                                       matrix::dense_matrix& weight_grad,
                                       feature_vec& bias_grad,
                                       bool compute_delta_prev) const {
    const auto& c = dynamic_cast<const gcn_cache&>(cache);
    const std::size_t num_nodes = delta_out.size();

    std::vector<feature_vec> delta_prev;
    if (compute_delta_prev) {
        delta_prev.assign(num_nodes, feature_vec(in_dim_, 0.0));
    }

    feature_vec delta_pre(out_dim_);
    feature_vec back(in_dim_);
    for (std::size_t i = 0; i < num_nodes; ++i) {
        // Through the activation
        for (std::size_t m = 0; m < out_dim_; ++m) {
            delta_pre[m] = delta_out[i][m] * activation_derivative(c.pre_act[i][m], activation_);
            bias_grad[m] += delta_pre[m];
        }

        for (std::size_t n = 0; n < in_dim_; ++n) {
            for (std::size_t m = 0; m < out_dim_; ++m) {
                weight_grad(n, m) += c.aggregate[i][n] * delta_pre[m];
            }
        }

        // d_loss/d_h_j = sum_i a_ij * W * delta_pre_i
        if (compute_delta_prev) {
            for (std::size_t n = 0; n < in_dim_; ++n) {
                double sum = 0.0;
                for (std::size_t m = 0; m < out_dim_; ++m) {
                    sum += delta_pre[m] * weights_(n, m);
                }

                back[n] = sum;
            }

            for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
                std::size_t j = adj.col_ind[k];
                double weight = adj.vals[k];
                for (std::size_t n = 0; n < in_dim_; ++n) {
                    delta_prev[j][n] += weight * back[n];
                }
            }
        }
    }

    return delta_prev;
}

} // namespace gnn
} // namespace gnnmath
