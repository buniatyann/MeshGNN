#ifndef GNNMATH_GNN_PIPELINE_HPP
#define GNNMATH_GNN_PIPELINE_HPP

#include "../mesh.hpp"
#include "../matrix.hpp"
#include "../vector.hpp"
#include "layer.hpp"

#include <memory>
#include <vector>

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
    std::vector<vector> process(const mesh::mesh& mesh) const;

    /// @brief Processes features directly with an adjacency matrix.
    /// @param features Input node features.
    /// @param adj Adjacency matrix.
    /// @return Output node features.
    /// @throws std::runtime_error If inputs are invalid.
    std::vector<vector> process(const std::vector<vector>& features,
                               const matrix::sparse_matrix& adj) const;

    /// @brief Returns the number of layers in the pipeline.
    /// @return Number of layers.
    std::size_t num_layers() const { return layers_.size(); }

    /// @brief Accesses layers for training.
    /// @return Const reference to layer pointers.
    const std::vector<std::unique_ptr<layer>>& layers() const { return layers_; }

private:
    std::vector<std::unique_ptr<layer>> layers_; ///< List of GNN layers.
};

} // namespace gnn
} // namespace gnnmath

#endif //GNNMATH_GNN_PIPELINE_HPP