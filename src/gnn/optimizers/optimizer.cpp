#include <gnnmath/gnn/optimizers/optimizer.hpp>
#include <gnnmath/gnn/optimizers/sgd.hpp>
#include <gnnmath/gnn/optimizers/adam.hpp>
#include <stdexcept>
#include <algorithm>

namespace gnnmath {
namespace gnn {

std::unique_ptr<optimizer> create_optimizer(const std::string& type,
                                            double learning_rate,
                                            double weight_decay) {
    std::string lower_type = type;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);

    if (lower_type == "sgd") {
        return std::make_unique<sgd_optimizer>(learning_rate, weight_decay);
    } 
    else if (lower_type == "adam") {
        return std::make_unique<adam_optimizer>(learning_rate, weight_decay);
    } 
    else {
        throw std::invalid_argument("Unknown optimizer type: " + type);
    }
}

} // namespace gnn
} // namespace gnnmath
