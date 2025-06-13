#ifndef GNNMATH_MESH_PROCESSOR_HPP
#define GNNMATH_MESH_PROCESSOR_HPP

#include "mesh.hpp"
#include "types.hpp"
#include <vector>

namespace gnnmath {
namespace mesh {

/**
 * @brief Computes quadric error approximation for edge collapse.
 * @param m Mesh object.
 * @param u First vertex index.
 * @param v Second vertex index.
 * @return Quadric error cost for collapsing edge (u, v).
 * @throws std::runtime_error If vertex indices are invalid.
 */
scalar_t compute_quadric_error(const mesh& m, index_t u, index_t v);

/**
 * @brief Simplifies the mesh using GNN-driven edge collapse.
 * @param m Mesh to simplify.
 * @param target_vertices Desired number of vertices.
 * @param gnn_scores Optional GNN-predicted edge collapse scores (one per edge).
 * @throws std::invalid_argument If target_vertices exceeds current vertex count or gnn_scores size is invalid.
 * @throws std::runtime_error If the mesh is invalid.
 */
void simplify_gnn_edge_collapse(mesh& m, index_t target_vertices,
                                const std::vector<scalar_t>& gnn_scores = {});

/**
 * @brief Simplifies the mesh by randomly removing vertices (for baseline testing).
 * @param m Mesh to simplify.
 * @param target_vertices Desired number of vertices.
 * @throws std::invalid_argument If target_vertices exceeds current vertex count.
 * @throws std::runtime_error If the mesh is invalid.
 */
void simplify_random_removal(mesh& m, index_t target_vertices);

}
}

#endif // GNNMATH_MESH_PROCESSOR_HPP