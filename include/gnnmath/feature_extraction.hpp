#ifndef GNNMATH_FEATURE_EXCRACTION_HPP
#define GNNMATH_FEATURE_EXCRACTION_HPP

#include "mesh.hpp"
#include "vector.hpp"
#include "types.hpp"
#include <vector>

namespace gnnmath {
namespace mesh {

/**
 * @brief Computes Gaussian curvature approximation for each vertex.
 * @param m Mesh object.
 * @return Vector of curvature values (one per vertex).
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<scalar_t> compute_gaussian_curvature(const mesh& m);

/**
 * @brief Computes combined node features (coordinates, normals, curvature).
 * @param m Mesh object.
 * @return Vector of feature vectors, each containing [x, y, z, nx, ny, nz, curvature].
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<vector::vector> compute_combined_node_features(const mesh& m);

/**
 * @brief Computes edge features based on neighbor properties (length, normal angle).
 * @param m Mesh object.
 * @return Vector of feature vectors, each containing [length, angle].
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<vector::vector> compute_neighbor_edge_features(const mesh& m);

} // namespace mesh
} // namespace gnnmath

#endif // GNNMATH_FEATURE_EXCRACTION_HPP