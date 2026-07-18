#include <gnnmath/gnn/layers/layer.hpp>
#include <gnnmath/math/vector.hpp>
#include <cmath>
#include <stdexcept>

namespace gnnmath {
namespace gnn {

feature_vec apply_activation(const feature_vec& v, activation_type act) {
    switch (act) {
        case activation_type::RELU:
            return gnnmath::vector::relu(v);
        case activation_type::MISH:
            return gnnmath::vector::mish(v);
        case activation_type::SIGMOID:
            return gnnmath::vector::sigmoid(v);
        case activation_type::GELU:
            return gnnmath::vector::gelu(v);
    }

    return v;
}

double activation_derivative(double x, activation_type act) {
    switch (act) {
        case activation_type::RELU:
            return x > 0.0 ? 1.0 : 0.0;

        case activation_type::SIGMOID: {
            double s = 1.0 / (1.0 + std::exp(-std::min(x, 700.0)));
            return s * (1.0 - s);
        }

        case activation_type::MISH: {
            // Mish: x * tanh(softplus(x))
            // Derivative: tanh(sp) + x * sech^2(sp) * sigmoid(x)
            double sp = std::log1p(std::exp(std::min(x, 700.0)));
            double tanh_sp = std::tanh(sp);
            double sig = 1.0 / (1.0 + std::exp(-std::min(x, 700.0)));
            double sech2 = 1.0 - tanh_sp * tanh_sp;
            return tanh_sp + x * sech2 * sig;
        }

        case activation_type::GELU: {
            // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
            // Derivative: 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
            constexpr double sqrt_2 = 1.4142135623730951;
            constexpr double sqrt_2pi = 2.5066282746310002;
            double erf_term = 0.5 * (1.0 + std::erf(x / sqrt_2));
            double exp_term = std::exp(-0.5 * x * x) / sqrt_2pi;
            return erf_term + x * exp_term;
        }

        default:
            return 1.0;
    }
}

std::vector<feature_vec> layer::forward_cached(const std::vector<feature_vec>& features,
                                         const matrix::sparse_matrix& adj,
                                         std::unique_ptr<layer_cache>& cache) const {
    cache.reset();
    return forward(features, adj);
}

std::vector<feature_vec> layer::backward(const std::vector<feature_vec>& /*delta_out*/,
                                   const std::vector<feature_vec>& /*input*/,
                                   const matrix::sparse_matrix& /*adj*/,
                                   const layer_cache& /*cache*/,
                                   matrix::dense_matrix& /*weight_grad*/,
                                   feature_vec& /*bias_grad*/,
                                   bool /*compute_delta_prev*/) const {
    throw std::runtime_error("layer::backward: not implemented for this layer type");
}

} // namespace gnn
} // namespace gnnmath
