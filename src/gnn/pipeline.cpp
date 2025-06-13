#include "../../include/gnnmath/gnn/pipeline.hpp"
#include <stdexcept>

namespace gnnmath {
namespace gnn {

void pipeline::add_layer(std::unique_ptr<layer> layer_ptr) {
    if (!layer_ptr) {
        throw std::runtime_error("add_layer: null layer");
    }
    if (!layers_.empty() && layers_.back()->out_features() != layer_ptr->in_features()) {
        throw std::runtime_error("add_layer: dimension mismatch");
    }

    layers_.push_back(std::move(layer_ptr));
}

std::vector<vector> pipeline::process(const mesh::mesh& mesh) const {
    if (layers_.empty()) {
        throw std::runtime_error("process: empty pipeline");
    }
    if (!mesh.is_valid()) {
        throw std::runtime_error("process: invalid mesh");
    }
    
    auto adj = mesh.to_adjacency_matrix();
    // Initial features: vertex coordinates
    std::vector<vector> features;
    const auto& vertices = mesh.vertices();
    for (const auto& v : vertices) {
        features.emplace_back(std::vector{v[0], v[1], v[2]});
    }

    return process(features, adj);
}

std::vector<vector> pipeline::process(const std::vector<vector>& features,
                                     const matrix::sparse_matrix& adj) const {
    if (layers_.empty()) {
        throw std::runtime_error("process: empty pipeline");
    }
    if (features.empty() || features.size() != adj.rows) {
        throw std::runtime_error("process: invalid input dimensions");
    }
    
    std::vector<vector> current_features = features;
    for (const auto& layer : layers_) {
        current_features = layer->forward(current_features, adj);
    }
    
    return current_features;
}

} // namespace gnn
} // namespace gnnmath