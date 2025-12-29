#include "../../include/gnnmath/gnn/pipeline.hpp"
#include <stdexcept>
#include <fstream>

namespace gnnmath {
namespace gnn {

// Use feature_vec from parent namespace
using gnnmath::feature_vec;

// Magic number and version for file format validation
static constexpr uint32_t MAGIC_NUMBER = 0x4D475050;  // "MGPP" - MeshGNN Pipeline
static constexpr uint32_t FORMAT_VERSION = 1;

void pipeline::add_layer(std::unique_ptr<layer> layer_ptr) {
    if (!layer_ptr) {
        throw std::runtime_error("add_layer: null layer");
    }
    if (!layers_.empty() && layers_.back()->out_features() != layer_ptr->in_features()) {
        throw std::runtime_error("add_layer: dimension mismatch");
    }

    layers_.push_back(std::move(layer_ptr));
}

std::vector<feature_vec> pipeline::process(const mesh::mesh& mesh) const {
    if (layers_.empty()) {
        throw std::runtime_error("process: empty pipeline");
    }
    if (!mesh.is_valid()) {
        throw std::runtime_error("process: invalid mesh");
    }

    auto adj = mesh.to_adjacency_matrix();
    // Initial features: vertex coordinates
    std::vector<feature_vec> features;
    const auto& vertices = mesh.vertices();
    for (const auto& v : vertices) {
        features.emplace_back(feature_vec{v[0], v[1], v[2]});
    }

    return process(features, adj);
}

std::vector<feature_vec> pipeline::process(const std::vector<feature_vec>& features,
                                     const matrix::sparse_matrix& adj) const {
    if (layers_.empty()) {
        throw std::runtime_error("process: empty pipeline");
    }
    if (features.empty() || features.size() != adj.rows) {
        throw std::runtime_error("process: invalid input dimensions");
    }

    std::vector<feature_vec> current_features = features;
    for (const auto& layer : layers_) {
        current_features = layer->forward(current_features, adj);
    }

    return current_features;
}

void pipeline::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("save: cannot open file for writing: " + filename);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(MAGIC_NUMBER));
    file.write(reinterpret_cast<const char*>(&FORMAT_VERSION), sizeof(FORMAT_VERSION));

    // Write number of layers
    uint32_t num_layers = static_cast<uint32_t>(layers_.size());
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    // Write each layer's parameters
    for (const auto& layer_ptr : layers_) {
        auto* gcn = dynamic_cast<gcn_layer*>(layer_ptr.get());
        auto* edge_conv = dynamic_cast<edge_conv_layer*>(layer_ptr.get());

        // Write layer type: 0 = unknown, 1 = GCN, 2 = EdgeConv
        uint8_t layer_type = 0;
        if (gcn) layer_type = 1;
        else if (edge_conv) layer_type = 2;
        file.write(reinterpret_cast<const char*>(&layer_type), sizeof(layer_type));

        if (gcn || edge_conv) {
            const matrix::dense_matrix& weights = gcn ? gcn->weights() : edge_conv->weights();
            const feature_vec& bias = gcn ? gcn->bias() : edge_conv->bias();

            // Write dimensions
            uint32_t rows = static_cast<uint32_t>(weights.rows());
            uint32_t cols = static_cast<uint32_t>(weights.cols());
            file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

            // Write weights
            for (std::size_t i = 0; i < weights.rows(); ++i) {
                for (std::size_t j = 0; j < weights.cols(); ++j) {
                    double val = weights(i, j);
                    file.write(reinterpret_cast<const char*>(&val), sizeof(val));
                }
            }

            // Write bias
            uint32_t bias_size = static_cast<uint32_t>(bias.size());
            file.write(reinterpret_cast<const char*>(&bias_size), sizeof(bias_size));
            for (double val : bias) {
                file.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
    }

    if (!file) {
        throw std::runtime_error("save: error writing to file: " + filename);
    }
}

void pipeline::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("load: cannot open file for reading: " + filename);
    }

    // Read and validate header
    uint32_t magic = 0;
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != MAGIC_NUMBER) {
        throw std::runtime_error("load: invalid file format (magic number mismatch)");
    }
    if (version != FORMAT_VERSION) {
        throw std::runtime_error("load: unsupported file version");
    }

    // Read number of layers
    uint32_t num_layers = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    if (num_layers != layers_.size()) {
        throw std::runtime_error("load: layer count mismatch (expected " +
                                 std::to_string(layers_.size()) + ", got " +
                                 std::to_string(num_layers) + ")");
    }

    // Read each layer's parameters
    for (std::size_t l = 0; l < num_layers; ++l) {
        auto* gcn = dynamic_cast<gcn_layer*>(layers_[l].get());
        auto* edge_conv = dynamic_cast<edge_conv_layer*>(layers_[l].get());

        uint8_t layer_type = 0;
        file.read(reinterpret_cast<char*>(&layer_type), sizeof(layer_type));

        // Validate layer type matches
        uint8_t expected_type = 0;
        if (gcn) expected_type = 1;
        else if (edge_conv) expected_type = 2;

        if (layer_type != expected_type) {
            throw std::runtime_error("load: layer type mismatch at layer " + std::to_string(l));
        }

        if (gcn || edge_conv) {
            matrix::dense_matrix& weights = gcn ? gcn->weights() : edge_conv->weights();
            feature_vec& bias = gcn ? gcn->bias() : edge_conv->bias();

            // Read dimensions
            uint32_t rows = 0, cols = 0;
            file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

            if (rows != weights.rows() || cols != weights.cols()) {
                throw std::runtime_error("load: weight dimension mismatch at layer " + std::to_string(l));
            }

            // Read weights
            for (std::size_t i = 0; i < weights.rows(); ++i) {
                for (std::size_t j = 0; j < weights.cols(); ++j) {
                    double val = 0.0;
                    file.read(reinterpret_cast<char*>(&val), sizeof(val));
                    weights(i, j) = val;
                }
            }

            // Read bias
            uint32_t bias_size = 0;
            file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));

            if (bias_size != bias.size()) {
                throw std::runtime_error("load: bias dimension mismatch at layer " + std::to_string(l));
            }

            for (std::size_t i = 0; i < bias.size(); ++i) {
                double val = 0.0;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                bias[i] = val;
            }
        }
    }

    if (!file) {
        throw std::runtime_error("load: error reading from file: " + filename);
    }
}

} // namespace gnn
} // namespace gnnmath
