#include <gnnmath/geometry/mesh.hpp>
#include <gnnmath/math/vector.hpp>
#include <gnnmath/core/random.hpp>
#include <gnnmath/geometry/feature_extraction.hpp>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <execution>

namespace gnnmath {
namespace mesh {

std::vector<scalar_t> compute_gaussian_curvature(const mesh& m) {
    m.validate();
    std::vector<scalar_t> curvature(m.n_vertices(), 0.0);

    // Vertex -> incident face indices
    std::vector<std::vector<index_t>> vertex_faces(m.n_vertices());
    // Face count per edge (indexed via the mesh's own edge map), to detect
    // boundary vertices
    std::vector<index_t> edge_face_count(m.n_edges(), 0);
    const auto& edge_map = m.edge_ind_map();
    for (index_t fi = 0; fi < m.n_faces(); ++fi) {
        const auto& f = m.faces()[fi];
        for (index_t c = 0; c < 3; ++c) {
            vertex_faces[f[c]].push_back(fi);
            index_t a = f[c], b = f[(c + 1) % 3];
            auto it = edge_map.find({std::min(a, b), std::max(a, b)});
            if (it != edge_map.end()) {
                ++edge_face_count[it->second];
            }
        }
    }

    // A vertex is on the boundary if any incident edge borders exactly one face
    std::vector<bool> boundary(m.n_vertices(), false);
    for (index_t e = 0; e < m.n_edges(); ++e) {
        if (edge_face_count[e] == 1) {
            boundary[m.edges()[e].first] = true;
            boundary[m.edges()[e].second] = true;
        }
    }

    for (index_t v = 0; v < m.n_vertices(); ++v) {
        if (vertex_faces[v].empty()) {
            continue;
        }

        scalar_t angle_sum = 0.0;
        scalar_t area_sum = 0.0;
        for (index_t fi : vertex_faces[v]) {
            const auto& f = m.faces()[fi];
            index_t v0 = f[0], v1 = f[1], v2 = f[2];
            // Find vertices u, w in triangle (v, u, w)
            index_t u = (v0 == v) ? v1 : (v1 == v) ? v2 : v0;
            index_t w = (v0 == v) ? v2 : (v1 == v) ? v0 : v1;
            // Compute angle at v
            auto vu = vector::operator-(m.vertices()[u], m.vertices()[v]);
            auto vw = vector::operator-(m.vertices()[w], m.vertices()[v]);
            scalar_t cos_theta = vector::dot_product(vu, vw) /
                (vector::euclidean_norm(vu) * vector::euclidean_norm(vw));
            if (cos_theta < -1.0) {
                cos_theta = -1.0;
            }

            if (cos_theta > 1.0) {
                cos_theta = 1.0;
            }

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

        if (area_sum > 1e-10) {
            // Angle deficit: 2*pi for interior vertices, pi on the boundary
            scalar_t full_angle = boundary[v] ? M_PI : 2.0 * M_PI;
            curvature[v] = (full_angle - angle_sum) / area_sum;
        }
    }

    return curvature;
}

std::vector<vector::vector> compute_combined_node_features(const mesh& m) {
    m.validate();
    auto coords = m.compute_node_features();
    auto normals = m.compute_normals();
    auto curvature = compute_gaussian_curvature(m);
    std::vector<vector::vector> features(m.n_vertices());
    // Indexed loop: a stateful counter in a parallel transform would race
    for (index_t i = 0; i < m.n_vertices(); ++i) {
        vector::vector feat(coords[i].begin(), coords[i].end());
        feat.insert(feat.end(), normals[i].begin(), normals[i].end());
        feat.push_back(curvature[i]);
        features[i] = std::move(feat);
    }

    return features;
}

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
                    if (cos_angle < -1.0) {
                        cos_angle = -1.0;
                    }
                    if (cos_angle > 1.0) {
                        cos_angle = 1.0;
                    }

                    scalar_t angle = std::acos(cos_angle);
                    return vector::vector{len, angle};
                });
                
    return features;
}

} // namespace mesh
} // namespace gnnmath