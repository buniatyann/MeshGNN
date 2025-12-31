#ifndef GNNMATH_GNN_PIPELINE_HPP
#define GNNMATH_GNN_PIPELINE_HPP

#include "../geometry/mesh.hpp"
#include "../math/dense_matrix.hpp"
#include "../math/sparse_matrix.hpp"
#include "../math/vector.hpp"
#include "layers/layer.hpp"
#include "layers/gcn_layer.hpp"
#include "layers/edge_conv_layer.hpp"

#include <memory>
#include <vector>
#include <string>
#include <fstream>

namespace gnnmath {
namespace gnn {

/// @brief GNN pipeline for stacking layers and processing meshes.
class pipeline {
public:
    /// @brief Constructs an empty pipeline.
    pipeline() = default;

    /// @brief Adds a layer to the pipeline.
    /// @param layer_ptr Unique pointer to a GNN layer.
    /// @throws std::runtime_error If layer is null or dimensions mismatch.
    void add_layer(std::unique_ptr<layer> layer_ptr);

    /// @brief Processes a mesh through the pipeline.
    /// @param mesh Input mesh.
    /// @return Node features after processing.
    /// @throws std::runtime_error If mesh is invalid or pipeline is empty.
    std::vector<feature_vec> process(const mesh::mesh& mesh) const;

    /// @brief Processes features directly with an adjacency matrix.
    /// @param features Input node features.
    /// @param adj Adjacency matrix.
    /// @return Output node features.
    /// @throws std::runtime_error If inputs are invalid.
    std::vector<feature_vec> process(const std::vector<feature_vec>& features,
                               const matrix::sparse_matrix& adj) const;

    /// @brief Returns the number of layers in the pipeline.
    /// @return Number of layers.
    std::size_t num_layers() const { return layers_.size(); }

    /// @brief Accesses layers for training.
    /// @return Const reference to layer pointers.
    const std::vector<std::unique_ptr<layer>>& layers() const { return layers_; }

    /// @brief Saves the pipeline weights to a binary file.
    /// @param filename Path to save the model.
    /// @throws std::runtime_error If file cannot be opened.
    void save(const std::string& filename) const;

    /// @brief Loads pipeline weights from a binary file.
    /// @param filename Path to load the model from.
    /// @throws std::runtime_error If file cannot be opened or format is invalid.
    void load(const std::string& filename);

private:
    std::vector<std::unique_ptr<layer>> layers_; ///< List of GNN layers.
};

} // namespace gnn
} // namespace gnnmath

#endif //GNNMATH_GNN_PIPELINE_HPP
