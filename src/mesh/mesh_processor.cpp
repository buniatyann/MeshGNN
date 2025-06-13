#include "../../include/gnnmath/mesh.hpp"
#include "../../include/gnnmath/vector.hpp"
#include "../../include/gnnmath/matrix.hpp"
#include "../../include/gnnmath//random.hpp"
#include <vector>
#include <queue>
#include <stdexcept>
#include <cmath>
#include <execution>

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
scalar_t compute_quadric_error(const mesh& m, index_t u, index_t v) {
    if (u >= m.n_vertices() || v >= m.n_vertices()) {
        throw std::runtime_error("compute_quadric_error: invalid vertex index");
    }
    // Use edge length as a proxy for quadric error (extend with full quadric matrices for accuracy)
    const auto& p0 = m.vertices()[u];
    const auto& p1 = m.vertices()[v];
    return vector::euclidean_norm(vector::operator-(p1, p0));
}

/**
 * @brief Simplifies the mesh using GNN-driven edge collapse.
 * @param m Mesh to simplify.
 * @param target_vertices Desired number of vertices.
 * @param gnn_scores Optional GNN-predicted edge collapse scores (one per edge).
 * @throws std::invalid_argument If target_vertices exceeds current vertex count or gnn_scores size is invalid.
 * @throws std::runtime_error If the mesh is invalid.
 */
void simplify_gnn_edge_collapse(mesh& m, index_t target_vertices,
                                const std::vector<scalar_t>& gnn_scores) {
    m.validate();
    if (target_vertices > m.n_vertices()) {
        throw std::invalid_argument("simplify_gnn_edge_collapse: target exceeds vertices");
    }
    if (!gnn_scores.empty() && gnn_scores.size() != m.n_edges()) {
        throw std::invalid_argument("simplify_gnn_edge_collapse: invalid gnn_scores size");
    }

    // Priority queue for edges based on collapse cost
    using cost_t = std::pair<scalar_t, index_t>;
    std::priority_queue<cost_t, std::vector<cost_t>, std::greater<cost_t>> pq;

    // Initialize costs using GNN scores or quadric error
    const auto& edges = m.edges();
    std::vector<scalar_t> costs(edges.size());
    if (!gnn_scores.empty()) {
        costs = gnn_scores;
    } else {
        std::transform(std::execution::par_unseq, edges.begin(), edges.end(), costs.begin(),
                       [&m](const auto& e) {
                           return compute_quadric_error(m, e.first, e.second);
                       });
    }

    // Populate queue using edge_ind_map for validation
    const auto& edge_map = m.edge_ind_map();
    for (index_t i = 0; i < edges.size(); ++i) {
        const auto& [u, v] = edges[i];
        auto key = std::make_pair(std::min(u, v), std::max(u, v));
        if (edge_map.count(key)) {
            pq.push({costs[i], i});
        }
    }

    // Track valid vertices and edges
    std::vector<bool> valid_vertices(m.n_vertices(), true);
    std::vector<bool> valid_edges(m.n_edges(), true);
    index_t current_vertices = m.n_vertices();

    while (current_vertices > target_vertices && !pq.empty()) {
        auto [cost, edge_idx] = pq.top();
        pq.pop();
        if (!valid_edges[edge_idx]) {
            continue;
        }
        auto [u, v] = edges[edge_idx];
        if (!valid_vertices[u] || !valid_vertices[v]) {
            continue;
        }

        // Collapse u -> v
        valid_vertices[u] = false;
        valid_edges[edge_idx] = false;
        --current_vertices;

        // Update vertex position to midpoint
        m.vertices()[v] = vector::scalar_multiply(
            vector::operator+(m.vertices()[u], m.vertices()[v]), 0.5);

        // Update faces, removing degenerates
        std::vector<mesh::face> new_faces;
        for (const auto& f : m.faces()) {
            mesh::face new_f = f;
            if (f[0] == u) new_f[0] = v;
            if (f[1] == u) new_f[1] = v;
            if (f[2] == u) new_f[2] = v;
            if (new_f[0] != new_f[1] && new_f[1] != new_f[2] && new_f[2] != new_f[0]) {
                new_faces.push_back(new_f);
            }
        }
        m.faces() = std::move(new_faces);

        // Rebuild edges and adjacency using edge_ind_map and adjacency
        m.edges().clear();
        m.edge_ind_map().clear();
        m.adjacency().clear();
        m.incident_edges().clear();
        for (const auto& [v0, v1, v2] : m.faces()) {
            std::vector<std::pair<index_t, index_t>> face_edges = {
                {std::min(v0, v1), std::max(v0, v1)},
                {std::min(v1, v2), std::max(v1, v2)},
                {std::min(v2, v0), std::max(v2, v0)}
            };
            for (const auto& [u2, v2] : face_edges) {
                auto key = std::make_pair(u2, v2);
                if (!m.edge_ind_map().count(key)) {
                    index_t edge_idx = m.edges().size();
                    m.edges().push_back({u2, v2});
                    m.edge_ind_map()[key] = edge_idx;
                    m.adjacency()[u2].push_back(v2);
                    m.adjacency()[v2].push_back(u2);
                    m.incident_edges()[u2].push_back(edge_idx);
                    m.incident_edges()[v2].push_back(edge_idx);
                }
            }
        }

        // Update costs for affected edges using incident_edges
        pq = std::priority_queue<cost_t, std::vector<cost_t>, std::greater<cost_t>>();
        valid_edges.assign(m.n_edges(), true);
        costs.resize(m.n_edges());
        std::transform(std::execution::par_unseq, m.edges().begin(), m.edges().end(), costs.begin(),
                       [&m](const auto& e) {
                           return compute_quadric_error(m, e.first, e.second);
                       });
        for (index_t i = 0; i < m.n_edges(); ++i) {
            const auto& [u2, v2] = m.edges()[i];
            if (valid_vertices[u2] && valid_vertices[v2]) {
                pq.push({costs[i], i});
            } else {
                valid_edges[i] = false;
            }
        }
    }

    // Compact vertices
    std::vector<mesh::vertex> new_vertices;
    std::vector<index_t> old_to_new(m.n_vertices(), 0);
    index_t new_idx = 0;
    for (index_t i = 0; i < m.n_vertices(); ++i) {
        if (valid_vertices[i]) {
            new_vertices.push_back(m.vertices()[i]);
            old_to_new[i] = new_idx++;
        }
    }
    m.vertices() = std::move(new_vertices);

    // Update faces and rebuild edges
    for (auto& f : m.faces()) {
        f[0] = old_to_new[f[0]];
        f[1] = old_to_new[f[1]];
        f[2] = old_to_new[f[2]];
    }
    m.edges().clear();
    m.edge_ind_map().clear();
    m.adjacency().clear();
    m.incident_edges().clear();
    for (const auto& [v0, v1, v2] : m.faces()) {
        std::vector<std::pair<index_t, index_t>> face_edges = {
            {std::min(v0, v1), std::max(v0, v1)},
            {std::min(v1, v2), std::max(v1, v2)},
            {std::min(v2, v0), std::max(v2, v0)}
        };
        for (const auto& [u2, v2] : face_edges) {
            auto key = std::make_pair(u2, v2);
            if (!m.edge_ind_map().count(key)) {
                index_t edge_idx = m.edges().size();
                m.edges().push_back({u2, v2});
                m.edge_ind_map()[key] = edge_idx;
                m.adjacency()[u2].push_back(v2);
                m.adjacency()[v2].push_back(u2);
                m.incident_edges()[u2].push_back(edge_idx);
                m.incident_edges()[v2].push_back(edge_idx);
            }
        }
    }

    m.validate();
}

/**
 * @brief Simplifies the mesh by randomly removing vertices (for baseline testing).
 * @param m Mesh to simplify.
 * @param target_vertices Desired number of vertices.
 * @throws std::invalid_argument If target_vertices exceeds current vertex count.
 * @throws std::runtime_error If the mesh is invalid.
 */
void simplify_random_removal(mesh& m, index_t target_vertices) {
    m.validate();
    if (target_vertices > m.n_vertices()) {
        throw std::invalid_argument("simplify_random_removal: target exceeds vertices");
    }

    // Sample vertices to remove using mesh::sample_vertices
    auto indices = m.sample_vertices(m.n_vertices() - target_vertices);
    std::vector<bool> valid_vertices(m.n_vertices(), true);
    for (index_t idx : indices) {
        valid_vertices[idx] = false;
    }

    // Update faces, keeping only those with valid vertices
    std::vector<mesh::face> new_faces;
    for (const auto& f : m.faces()) {
        if (valid_vertices[f[0]] && valid_vertices[f[1]] && valid_vertices[f[2]]) {
            new_faces.push_back(f);
        }
    }
    m.faces() = std::move(new_faces);

    // Compact vertices
    std::vector<mesh::vertex> new_vertices;
    std::vector<index_t> old_to_new(m.n_vertices(), 0);
    index_t new_idx = 0;
    for (index_t i = 0; i < m.n_vertices(); ++i) {
        if (valid_vertices[i]) {
            new_vertices.push_back(m.vertices()[i]);
            old_to_new[i] = new_idx++;
        }
    }
    m.vertices() = std::move(new_vertices);

    // Update faces with new indices and rebuild edges
    for (auto& f : m.faces()) {
        f[0] = old_to_new[f[0]];
        f[1] = old_to_new[f[1]];
        f[2] = old_to_new[f[2]];
    }
    m.edges().clear();
    m.edge_ind_map().clear();
    m.adjacency().clear();
    m.incident_edges().clear();
    for (const auto& [v0, v1, v2] : m.faces()) {
        std::vector<std::pair<index_t, index_t>> face_edges = {
            {std::min(v0, v1), std::max(v0, v1)},
            {std::min(v1, v2), std::max(v1, v2)},
            {std::min(v2, v0), std::max(v2, v0)}
        };
        for (const auto& [u2, v2] : face_edges) {
            auto key = std::make_pair(u2, v2);
            if (!m.edge_ind_map().count(key)) {
                index_t edge_idx = m.edges().size();
                m.edges().push_back({u2, v2});
                m.edge_ind_map()[key] = edge_idx;
                m.adjacency()[u2].push_back(v2);
                m.adjacency()[v2].push_back(u2);
                m.incident_edges()[u2].push_back(edge_idx);
                m.incident_edges()[v2].push_back(edge_idx);
            }
        }
    }

    m.validate();
}

} // namespace mesh
} // namespace gnnmath