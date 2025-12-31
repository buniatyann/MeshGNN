#include <gnnmath/gnn/losses/loss.hpp>
#include <gnnmath/gnn/losses/mse.hpp>
#include <gnnmath/gnn/losses/cross_entropy.hpp>
#include <stdexcept>
#include <algorithm>

namespace gnnmath {
namespace gnn {

std::unique_ptr<loss_function> create_loss(const std::string& type) {
    std::string lower_type = type;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);

    if (lower_type == "mse" || lower_type == "mean_squared_error") {
        return std::make_unique<mse_loss>();
    } 
    else if (lower_type == "cross_entropy" || lower_type == "ce") {
        return std::make_unique<cross_entropy_loss>();
    } 
    else {
        throw std::invalid_argument("Unknown loss type: " + type);
    }
}

} // namespace gnn
} // namespace gnnmath
