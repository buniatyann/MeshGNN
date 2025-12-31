#include <gnnmath/math/vector.hpp>
#include <cmath>
#include <algorithm>
#include <execution>

namespace gnnmath {
namespace vector {
    vector operator+(const vector& a, const vector& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Vector addition: sizes " + std::to_string(a.size()) + 
                                    " and " + std::to_string(b.size()) + " do not match");
        }
        
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(), ans.begin(),
                       [](double x, double y) { return x + y; });
        
        return ans;
    }

    vector& operator+=(vector& a, const vector& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Vector addition: sizes " + std::to_string(a.size()) + 
                                    " and " + std::to_string(b.size()) + " do not match");
        }
        
        std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(), a.begin(),
                       [](double x, double y) { return x + y; });
        
        return a;
    }

    vector operator-(const vector& a, const vector& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Vector subtraction: sizes " + std::to_string(a.size()) + 
                                    " and " + std::to_string(b.size()) + " do not match");
        }
        
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(), ans.begin(),
                       [](double x, double y) { return x - y; });
        
        return ans;
    }

    vector& operator-=(vector& a, const vector& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Vector subtraction: sizes " + std::to_string(a.size()) + 
                                    " and " + std::to_string(b.size()) + " do not match");
        }
        
        std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(), a.begin(),
                       [](double x, double y) { return x - y; });
        
        return a;
    }

    vector scalar_multiply(const vector& a, double b) {
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), ans.begin(),
                       [b](double x) { return x * b; });
        
        return ans;
    }

    double dot_product(const vector& a, const vector& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Dot product: sizes " + std::to_string(a.size()) + 
                                    " and " + std::to_string(b.size()) + " do not match");
        }
        
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }

    double euclidean_norm(const vector& a) {
        double squared_sum = 0.0;
        for (double x : a) {
            squared_sum += x * x;
        }
        
        return std::sqrt(squared_sum);
    }

    vector relu(const vector& a) {
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), ans.begin(),
                       [](double x) { return std::max(0.0, x); });
        
        return ans;
    }

    vector sigmoid(const vector& a) {
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), ans.begin(),
                       [](double x) {
                           // Clamp x to avoid overflow and ensure result stays strictly in (0, 1)
                           // The threshold is chosen so that the computation doesn't round to 0 or 1
                           // For x > 36.7, 1 - exp(-x) rounds to exactly 1.0
                           // For x < -36.7, exp(x) rounds to exactly 0.0
                           constexpr double threshold = 36.7;

                           if (x > threshold) {
                               // Return largest double strictly less than 1.0
                               return std::nextafter(1.0, 0.0);
                           } 
                           else if (x < -threshold) {
                               // Return smallest positive double
                               return std::nextafter(0.0, 1.0);
                           }
                           
                           return 1.0 / (1.0 + std::exp(-x));
                       });

        return ans;
    }

    vector mish(const vector& a) {
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), ans.begin(),
                       [](double x) { return x * std::tanh(std::log1p(std::exp(std::min(x, 700.0)))); });
        
        return ans;
    }

    vector softmax(const vector& a) {
        if (a.empty()) {
            throw std::runtime_error("Softmax: input vector is empty");
        }
        vector ans(a.size());
        // Compute exp(x) for each element, clamping to avoid overflow
        std::transform(std::execution::par_unseq, a.begin(), a.end(), ans.begin(),
                       [](double x) { return std::exp(std::min(x, 700.0)); });
        // Compute sum of exp(x)
        double sum = std::accumulate(ans.begin(), ans.end(), 0.0);
        constexpr double epsilon = 1e-10;
        if (sum < epsilon) {
            throw std::runtime_error("Softmax: sum of exponentials is too small");
        }
        
        std::transform(std::execution::par_unseq, ans.begin(), ans.end(), ans.begin(),
                       [sum](double x) { return x / sum; });
        
        return ans;
    }

    vector softplus(const vector& a) {
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), ans.begin(),
                       [](double x) { return std::log1p(std::exp(std::min(x, 700.0))); });
        
        return ans;
    }

    vector gelu(const vector& a) {
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), ans.begin(),
                       [](double x) {
                           // Approximate GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
                           constexpr double sqrt_2 = 1.4142135623730951;
                           return x * 0.5 * (1.0 + std::erf(x / sqrt_2));
                       });
        
        return ans;
    }

    vector silu(const vector& a) {
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), ans.begin(),
                       [](double x) { return x / (1.0 + std::exp(-std::min(x, 700.0))); });
        
        return ans;
    }

    vector softsign(const vector& a) {
        vector ans(a.size());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), ans.begin(),
                       [](double x) { return x / (1.0 + std::abs(x)); });
        
        return ans;
    }
} // namespace vector
} // namespace gnnmath
