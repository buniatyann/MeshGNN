#include <gnnmath/gnn/pipeline.hpp>
#include <stdexcept>
#include <fstream>

namespace gnnmath {
namespace gnn {

using gnnmath::feature_vec;

// Magic number and version for file format validation.
// v1: per layer {type u8, rows u32, cols u32, weights, bias_size u32, bias};
//     requires a prebuilt pipeline of matching shape to load into.
// v2: per layer {type u8, activation u8, in_dim u32, out_dim u32, weights, bias};
//     self-describing, so load() can reconstruct the pipeline from scratch.
//     The activation byte freezes the enum order RELU=0, MISH=1, SIGMOID=2, GELU=3.
static constexpr uint32_t MAGIC_NUMBER = 0x4D475050;  // "MGPP" - MeshGNN Pipeline
static constexpr uint32_t FORMAT_VERSION = 2;

void pipeline::add_layer(std::unique_ptr<layer> layer_ptr) {
    if (!layer_ptr) {
        throw std::runtime_error("add_layer: null layer");
    }
    if (!layers_.empty() && layers_.back()->out_features() != layer_ptr->in_features()) {
        throw std::runtime_error("add_layer: dimension mismatch");
    }

    layers_.push_back(std::move(layer_ptr));
}

std::vector<feature_vec> pipeline::process(const mesh::mesh& mesh, bool normalize_adj) const {
    if (layers_.empty()) {
        throw std::runtime_error("process: empty pipeline");
    }
    if (!mesh.is_valid()) {
        throw std::runtime_error("process: invalid mesh");
    }

    auto adj = mesh.to_adjacency_matrix();
    if (normalize_adj) {
        adj = matrix::normalized_adjacency(adj);
    }
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
    for (const auto& layer_ptr : layers_) {
        // Layer type byte: 0 = unknown, 1 = GCN, 2 = EdgeConv
        uint8_t layer_type = static_cast<uint8_t>(layer_ptr->kind());
        uint8_t activation = static_cast<uint8_t>(layer_ptr->act());
        uint32_t in_dim = static_cast<uint32_t>(layer_ptr->in_features());
        uint32_t out_dim = static_cast<uint32_t>(layer_ptr->out_features());
        file.write(reinterpret_cast<const char*>(&layer_type), sizeof(layer_type));
        file.write(reinterpret_cast<const char*>(&activation), sizeof(activation));
        file.write(reinterpret_cast<const char*>(&in_dim), sizeof(in_dim));
        file.write(reinterpret_cast<const char*>(&out_dim), sizeof(out_dim));

        param_refs params = layer_ptr->parameters();
        if (layer_type != 0 && params.weights && params.bias) {
            // Weights are in_dim x out_dim and bias is out_dim for all known
            // layer types, so no separate dimension fields are needed
            const matrix::dense_matrix& weights = *params.weights;
            for (std::size_t i = 0; i < weights.rows(); ++i) {
                for (std::size_t j = 0; j < weights.cols(); ++j) {
                    double val = weights(i, j);
                    file.write(reinterpret_cast<const char*>(&val), sizeof(val));
                }
            }

            for (double val : *params.bias) {
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
    if (version != 1 && version != 2) {
        throw std::runtime_error("load: unsupported file version " + std::to_string(version));
    }

    uint32_t num_layers = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    // v2 files are self-describing: an empty pipeline is reconstructed from
    // the file; a prebuilt pipeline is validated against it.
    const bool reconstruct = layers_.empty();
    if (reconstruct && version == 1) {
        throw std::runtime_error("load: v1 files require a prebuilt pipeline of matching shape");
    }
    if (!reconstruct && num_layers != layers_.size()) {
        throw std::runtime_error("load: layer count mismatch (expected " +
                                 std::to_string(layers_.size()) + ", got " +
                                 std::to_string(num_layers) + ")");
    }

    auto read_params = [&file, &filename](matrix::dense_matrix& weights, feature_vec& bias) {
        for (std::size_t i = 0; i < weights.rows(); ++i) {
            for (std::size_t j = 0; j < weights.cols(); ++j) {
                double val = 0.0;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                weights(i, j) = val;
            }
        }

        for (std::size_t i = 0; i < bias.size(); ++i) {
            double val = 0.0;
            file.read(reinterpret_cast<char*>(&val), sizeof(val));
            bias[i] = val;
        }

        if (!file) {
            throw std::runtime_error("load: error reading from file: " + filename);
        }
    };

    for (std::size_t l = 0; l < num_layers; ++l) {
        uint8_t layer_type = 0;
        file.read(reinterpret_cast<char*>(&layer_type), sizeof(layer_type));

        if (version == 1) {
            // v1 record: type u8, rows u32, cols u32, weights, bias_size u32, bias
            uint8_t expected_type = static_cast<uint8_t>(layers_[l]->kind());
            if (layer_type != expected_type) {
                throw std::runtime_error("load: layer type mismatch at layer " + std::to_string(l));
            }

            param_refs params = layers_[l]->parameters();
            if (params.weights && params.bias) {
                uint32_t rows = 0, cols = 0;
                file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
                file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
                if (rows != params.weights->rows() || cols != params.weights->cols()) {
                    throw std::runtime_error("load: weight dimension mismatch at layer " + std::to_string(l));
                }

                // Read weights, then the bias size field, then the bias
                for (std::size_t i = 0; i < rows; ++i) {
                    for (std::size_t j = 0; j < cols; ++j) {
                        double val = 0.0;
                        file.read(reinterpret_cast<char*>(&val), sizeof(val));
                        (*params.weights)(i, j) = val;
                    }
                }

                uint32_t bias_size = 0;
                file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));
                if (bias_size != params.bias->size()) {
                    throw std::runtime_error("load: bias dimension mismatch at layer " + std::to_string(l));
                }

                for (std::size_t i = 0; i < bias_size; ++i) {
                    double val = 0.0;
                    file.read(reinterpret_cast<char*>(&val), sizeof(val));
                    (*params.bias)[i] = val;
                }
            }

            continue;
        }

        // v2 record: type u8, activation u8, in_dim u32, out_dim u32, weights, bias
        uint8_t activation = 0;
        uint32_t in_dim = 0, out_dim = 0;
        file.read(reinterpret_cast<char*>(&activation), sizeof(activation));
        file.read(reinterpret_cast<char*>(&in_dim), sizeof(in_dim));
        file.read(reinterpret_cast<char*>(&out_dim), sizeof(out_dim));
        if (activation > static_cast<uint8_t>(activation_type::GELU)) {
            throw std::runtime_error("load: invalid activation type at layer " + std::to_string(l));
        }

        activation_type act = static_cast<activation_type>(activation);
        if (reconstruct) {
            switch (static_cast<layer_kind>(layer_type)) {
                case layer_kind::gcn:
                    add_layer(std::make_unique<gcn_layer>(in_dim, out_dim, act));
                    break;
                case layer_kind::edge_conv:
                    add_layer(std::make_unique<edge_conv_layer>(in_dim, out_dim, act));
                    break;
                default:
                    throw std::runtime_error("load: cannot reconstruct unknown layer type at layer " +
                                             std::to_string(l));
            }
        }
        else {
            if (layer_type != static_cast<uint8_t>(layers_[l]->kind()) ||
                act != layers_[l]->act() ||
                in_dim != layers_[l]->in_features() || out_dim != layers_[l]->out_features()) {
                throw std::runtime_error("load: layer description mismatch at layer " + std::to_string(l));
            }
        }

        if (layer_type != 0) {
            param_refs params = layers_[l]->parameters();
            if (!params.weights || !params.bias) {
                throw std::runtime_error("load: layer " + std::to_string(l) + " has no parameters");
            }

            read_params(*params.weights, *params.bias);
        }
    }

    if (!file) {
        throw std::runtime_error("load: error reading from file: " + filename);
    }
}

pipeline pipeline::load_new(const std::string& filename) {
    pipeline p;
    p.load(filename);
    return p;
}

} // namespace gnn
} // namespace gnnmath
