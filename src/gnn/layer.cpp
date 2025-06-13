#include "../../include/gnnmath/gnn/layer.hpp"
#include "../../include/gnnmath/random.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gnnmath {
namespace gnn {

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

std::vector<vector> gcn_layer::forward(const std::vector<vector>& features,
                                      const matrix::sparse_matrix& adj) const {
    if (features.empty() || features.size() != adj.rows || (features[0].size() != in_dim_)) {
        throw std::runtime_error("gcn_layer forward: dimension mismatch");
    }
    
    std::vector<vector> out(features.size(), vector(out_dim_, 0.0));
    for (std::size_t i = 0; i < features.size(); ++i) {
        vector row = adj.multiply(features[i]);
        for (std::size_t j = 0; j < out_dim_; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < in_dim_; ++k) {
                sum += row[k] * weights_(k, j);
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
                row = vector::relu(row);
                break;
            case activation_type::MISH:
                row = vector::mish(row);
                break;
            case activation_type::SIGMOID:
                row = vector::sigmoid(row);
                break;
            case activation_type::GELU:
                row = vector::gelu(row);
                break;
        }
    }

    return out;
}

edge_conv_layer::edge_conv_layer(std::size_t in_dim, std::size_t out_dim, activation_type activation)
    : in_dim_(in_dim), out_dim_(out_dim), weights_(in_dim, out_dim), bias_(out_dim), activation_(activation) {
    if (in_dim == 0 || out_dim == 0) {
        throw std::runtime_error("edge_conv_layer: dimensions must be non-zero");
    }
    
    auto data = random::uniform_vector(in_dim * out_dim, -0.1, 0.1);
    for (std::size_t i = 0; i < in_dim; ++i) {
        for (std::size_t j = 0; j < out_dim; ++j) {
            weights_(i, j) = data[i * out_dim + j];
        }
    }
}

std::vector<vector> edge_conv_layer::forward(const std::vector<vector>& features,
                                            const matrix::sparse_matrix& adj) const {
    if (features.empty() || features.size() != adj.rows || (features[0].size() != in_dim_)) {
        throw std::runtime_error("edge_conv_layer forward: dimension mismatch");
    }
    
    std::vector<vector> out(features.size(), vector(out_dim_, 0.0));
    for (std::size_t i = 0; i < features.size(); ++i) {
        for (std::size_t k = adj.row_ptr[i]; k < adj.row_ptr[i + 1]; ++k) {
            std::size_t j = adj.col_ind[k];
            vector diff = vector::operator-(features[i], features[j]);
            vector weighted = vector(out_dim_, 0.0);
            for (std::size_t m = 0; m < out_dim_; ++m) {
                double sum = 0.0;
    
                for (std::size_t n = 0; n < in_dim_; ++n) {
                    sum += diff[n] * weights_(n, m);
                }
    
                weighted[m] = sum;
            }
    
            switch (activation_) {
                case activation_type::RELU:
                    weighted = vector::relu(weighted);
                    break;
                case activation_type::MISH:
                    weighted = vector::mish(weighted);
                    break;
                case activation_type::SIGMOID:
                    weighted = vector::sigmoid(weighted);
                    break;
                case activation_type::GELU:
                    weighted = vector::gelu(weighted);
                    break;
            }
    
            for (std::size_t m = 0; m < out_dim_; ++m) {
                out[i][m] += weighted[m];
            }
        }
    
        out[i] = vector::operator+(out[i], bias_);
        if (!matrix::is_valid(matrix::dense_matrix({out[i]}))) {
            throw std::runtime_error("edge_conv_layer forward: non-finite result");
        }
    }
    
    return out;
}

} // namespace gnn
} // namespace gnnmath