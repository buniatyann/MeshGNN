#include "../../include/gnnmath/mesh.hpp"
#include "../../include/gnnmath/vector.hpp"
#include "../../include/gnnmath/random.hpp"
#include "../../include/gnnmath/feature_extraction.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <execution>

namespace gnnmath {
namespace mesh {

/**
 * @brief Computes Gaussian curvature approximation for each vertex.
 * @param m Mesh object.
 * @return Vector of curvature values (one per vertex).
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<scalar_t> compute_gaussian_curvature(const mesh& m) {
    m.validate();
    std::vector<scalar_t> curvature(m.n_vertices(), 0.0);
    const auto& normals = m.compute_normals();
    const auto& adj = m.adjacency();

    for (index_t v = 0; v < m.n_vertices(); ++v) {
        auto neighbors = adj.count(v) ? adj.at(v) : std::vector<index_t>{};
        if (neighbors.empty()) {
            continue;
        }
        scalar_t angle_sum = 0.0;
        scalar_t area_sum = 0.0;

        // Iterate over incident triangles
        for (const auto& f : m.faces()) {
            if (f[0] == v || f[1] == v || f[2] == v) {
                index_t v0 = f[0], v1 = f[1], v2 = f[2];
                // Find vertices u, w in triangle (v, u, w)
                index_t u = (v0 == v) ? v1 : (v1 == v) ? v2 : v0;
                index_t w = (v0 == v) ? v2 : (v1 == v) ? v0 : v1;
                // Compute angle at v
                auto vu = vector::operator-(m.vertices()[u], m.vertices()[v]);
                auto vw = vector::operator-(m.vertices()[w], m.vertices()[v]);
                scalar_t cos_theta = vector::dot_product(vu, vw) /
                    (vector::euclidean_norm(vu) * vector::euclidean_norm(vw));
                if (cos_theta < -1.0) cos_theta = -1.0;
                if (cos_theta > 1.0) cos_theta = 1.0;
                scalar_t theta = std::acos(cos_theta);
                angle_sum += theta;
                // Approximate area as 1/3 of triangle area
                auto cross = vector::vector{
                    vu[1] * vw[2] - vu[2] * vw[1],
                    vu[2] * vw[0] - vu[0] * vw[2],
                    vu[0] * vw[1] - vu[1] * vw[0]
                };
                area_sum += vector::euclidean_norm(cross) / 6.0; // 1/2 for triangle, 1/3 for vertex
            }
        }

        if (area_sum > 1e-10) {
            curvature[v] = (2.0 * M_PI - angle_sum) / area_sum;
        }
    }

    return curvature;
}

/**
 * @brief Computes combined node features (coordinates, normals, curvature).
 * @param m Mesh object.
 * @return Vector of feature vectors, each containing [x, y, z, nx, ny, nz, curvature].
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<vector::vector> compute_combined_node_features(const mesh& m) {
    m.validate();
    auto coords = m.compute_node_features();
    auto normals = m.compute_normals();
    auto curvature = compute_gaussian_curvature(m);

    std::vector<vector::vector> features(m.n_vertices());
    std::transform(std::execution::par_unseq, coords.begin(), coords.end(), features.begin(),
                [&normals, &curvature, i = 0](const auto& coord) mutable {
                    vector::vector feat(coord.begin(), coord.end());
                    feat.insert(feat.end(), normals[i].begin(), normals[i].end());
                    feat.push_back(curvature[i++]);
                    return feat;
                });
    return features;
}

/**
 * @brief Computes edge features based on neighbor properties (length, normal angle).
 * @param m Mesh object.
 * @return Vector of feature vectors, each containing [length, angle].
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<vector::vector> compute_neighbor_edge_features(const mesh& m) {
    m.validate();
    const auto& edges = m.edges();
    const auto& edge_map = m.edge_ind_map();
    auto normals = m.compute_normals();
    std::vector<vector::vector> features(edges.size());

    std::transform(std::execution::par_unseq, edges.begin(), edges.end(), features.begin(),
                [&m, &normals, &edge_map](const auto& e) {
                    const auto& [u, v] = e;
                    auto key = std::make_pair(std::min(u, v), std::max(u, v));
                    if (!edge_map.count(key)) {
                        return vector::vector{0.0, 0.0}; // Invalid edge
                    }
                    // Edge length
                    scalar_t len = vector::euclidean_norm(
                        vector::operator-(m.vertices()[u], m.vertices()[v]));
                    // Normal angle difference
                    scalar_t cos_angle = vector::dot_product(normals[u], normals[v]);
                    if (cos_angle < -1.0) cos_angle = -1.0;
                    if (cos_angle > 1.0) cos_angle = 1.0;
                    scalar_t angle = std::acos(cos_angle);
                    return vector::vector{len, angle};
                });
    return features;
}

} // namespace mesh
} // namespace gnnmath