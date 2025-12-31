#include <gnnmath/gnn/layers/gcn_layer.hpp>
#include <gnnmath/core/random.hpp>
#include <gnnmath/math/vector.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gnnmath {
namespace gnn {

using gnnmath::feature_vec;

gcn_layer::gcn_layer(std::size_t in_dim, std::size_t out_dim, activation_type activation)
    : in_dim_(in_dim), out_dim_(out_dim), weights_(in_dim, out_dim), bias_(out_dim), activation_(activation) {
    if (in_dim == 0 || out_dim == 0) {
        throw std::runtime_error("gcn_layer: dimensions must be non-zero");
    }

    auto data = random::uniform_vector(in_dim * out_dim, -0.1, 0.1);
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

} // namespace gnn
} // namespace gnnmath
