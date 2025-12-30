#include <gnnmath/geometry/mesh.hpp>
#include <gnnmath/math/vector.hpp>
#include <gnnmath/math/dense_matrix.hpp>
#include <gnnmath/math/sparse_matrix.hpp>
#include <gnnmath/core/random.hpp>
#include <vector>
#include <queue>
#include <set>
#include <stdexcept>
#include <cmath>
#include <execution>

namespace gnnmath {
namespace mesh {

scalar_t compute_quadric_error(const mesh& m, index_t u, index_t v) {
    if (u >= m.n_vertices() || v >= m.n_vertices()) {
        throw std::runtime_error("compute_quadric_error: invalid vertex index");
    }
    // Use edge length as a proxy for quadric error (extend with full quadric matrices for accuracy)
    const auto& p0 = m.vertices()[u];
    const auto& p1 = m.vertices()[v];

    return vector::euclidean_norm(vector::operator-(p1, p0));
}

void simplify_gnn_edge_collapse(mesh& m, index_t target_vertices,
                                const std::vector<scalar_t>& gnn_scores) {
    m.validate();
    if (target_vertices > m.n_vertices()) {
        throw std::invalid_argument("simplify_gnn_edge_collapse: target exceeds vertices");
    }
    if (!gnn_scores.empty() && gnn_scores.size() != m.n_edges()) {
        throw std::invalid_argument("simplify_gnn_edge_collapse: invalid gnn_scores size");
    }

    // Priority queue: (cost, edge_idx, version) - for lazy deletion
    using pq_entry = std::tuple<scalar_t, index_t, index_t>;
    std::priority_queue<pq_entry, std::vector<pq_entry>, std::greater<pq_entry>> pq;

    std::vector<scalar_t> costs(m.n_edges());
    std::vector<index_t> edge_versions(m.n_edges(), 0);

    std::vector<bool> valid_vertices(m.n_vertices(), true);
    std::vector<bool> valid_edges(m.n_edges(), true);

    // costs initialization
    const auto& edges = m.edges();
    if (!gnn_scores.empty()) {
        costs = gnn_scores;
    } 
    else {
        for (index_t i = 0; i < edges.size(); ++i) {
            costs[i] = compute_quadric_error(m, edges[i].first, edges[i].second);
        }
    }

    for (index_t i = 0; i < edges.size(); ++i) {
        pq.push({costs[i], i, 0});
    }

    index_t current_vertices = m.n_vertices();
    while (current_vertices > target_vertices && !pq.empty()) {
        auto [cost, edge_idx, version] = pq.top();
        pq.pop();

        // Skip if edge was invalidated or version is stale
        if (!valid_edges[edge_idx] || version != edge_versions[edge_idx]) {
            continue;
        }

        auto [u, v] = m.edges()[edge_idx];
        if (!valid_vertices[u] || !valid_vertices[v]) {
            continue;
        }

        // Collapse u -> v (u is removed, v survives at midpoint)
        valid_vertices[u] = false;
        valid_edges[edge_idx] = false;
        --current_vertices;

        // Update vertex position to midpoint
        m.vertices()[v] = vector::scalar_multiply(
            vector::operator+(m.vertices()[u], m.vertices()[v]), 0.5);

        // Collect edges that need cost updates (edges incident to v)
        std::set<index_t> edges_to_update;

        // Get incident edges of u and transfer them to v
        if (m.incident_edges().count(u)) {
            for (index_t inc_edge : m.incident_edges().at(u)) {
                if (valid_edges[inc_edge] && inc_edge != edge_idx) {
                    auto& [eu, ev] = m.edges()[inc_edge];
                    // Redirect edge from u to v
                    if (eu == u) {
                        eu = v;
                    }
                    if (ev == u) {
                        ev = v;
                    }

                    // Check if edge became degenerate (self-loop)
                    if (eu == ev) {
                        valid_edges[inc_edge] = false;
                    } 
                    else {
                        edges_to_update.insert(inc_edge);
                    }
                }
            }
        }

        // Add edges incident to v for update
        if (m.incident_edges().count(v)) {
            for (index_t inc_edge : m.incident_edges().at(v)) {
                if (valid_edges[inc_edge]) {
                    edges_to_update.insert(inc_edge);
                }
            }
        }

        // Update costs and re-add to priority queue with new version
        for (index_t eidx : edges_to_update) {
            if (!valid_edges[eidx]) {
                continue;
            }

            auto [eu, ev] = m.edges()[eidx];
            if (!valid_vertices[eu] || !valid_vertices[ev]) {
                valid_edges[eidx] = false;
                continue;
            }

            // new cost
            costs[eidx] = compute_quadric_error(m, eu, ev);
            edge_versions[eidx]++;
            pq.push({costs[eidx], eidx, edge_versions[eidx]});
        }
    }

    // rebuild mesh with only valid elements
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

    // remap indices and filter degenerate faces
    std::vector<mesh::face> new_faces;
    for (const auto& f : m.faces()) {
        if (!valid_vertices[f[0]] || !valid_vertices[f[1]] || !valid_vertices[f[2]]) {
            continue;
        }
        
        mesh::face new_f = {old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]};
        // Check for degenerate face
        if (new_f[0] != new_f[1] && new_f[1] != new_f[2] && new_f[2] != new_f[0]) {
            new_faces.push_back(new_f);
        }
    }

    m.faces() = std::move(new_faces);
    // Rebuild edge structures from faces
    m.edges().clear();
    m.edge_ind_map().clear();
    m.adjacency().clear();
    m.incident_edges().clear();

    for (const auto& [v0, v1, v2] : m.faces()) {
        std::array<std::pair<index_t, index_t>, 3> face_edges = {{
            {std::min(v0, v1), std::max(v0, v1)},
            {std::min(v1, v2), std::max(v1, v2)},
            {std::min(v2, v0), std::max(v2, v0)}
        }};
        
        for (const auto& [eu, ev] : face_edges) {
            auto key = std::make_pair(eu, ev);
            if (!m.edge_ind_map().count(key)) {
                index_t eidx = m.edges().size();
                m.edges().push_back({eu, ev});
                m.edge_ind_map()[key] = eidx;
                m.adjacency()[eu].push_back(ev);
                m.adjacency()[ev].push_back(eu);
                m.incident_edges()[eu].push_back(eidx);
                m.incident_edges()[ev].push_back(eidx);
            }
        }
    }

    m.validate();
}


void simplify_random_removal(mesh& m, index_t target_vertices) {
    m.validate();
    if (target_vertices > m.n_vertices()) {
        throw std::invalid_argument("simplify_random_removal: target exceeds vertices");
    }

    // Sample vertices to remove
    auto indices = m.sample_vertices(m.n_vertices() - target_vertices);
    std::vector<bool> valid_vertices(m.n_vertices(), true);
    for (index_t idx : indices) {
        valid_vertices[idx] = false;
    }

    // keep only faces with all valid vertices
    std::vector<mesh::face> new_faces;
    new_faces.reserve(m.n_faces());
    for (const auto& f : m.faces()) {
        if (valid_vertices[f[0]] && valid_vertices[f[1]] && valid_vertices[f[2]]) {
            new_faces.push_back(f);
        }
    }

    m.faces() = std::move(new_faces);

    // Compact vertices
    std::vector<mesh::vertex> new_vertices;
    new_vertices.reserve(target_vertices);
    std::vector<index_t> old_to_new(m.n_vertices(), 0);
    index_t new_idx = 0;
    for (index_t i = 0; i < m.n_vertices(); ++i) {
        if (valid_vertices[i]) {
            new_vertices.push_back(m.vertices()[i]);
            old_to_new[i] = new_idx++;
        }
    }

    m.vertices() = std::move(new_vertices);
    // Update face indices
    for (auto& f : m.faces()) {
        f[0] = old_to_new[f[0]];
        f[1] = old_to_new[f[1]];
        f[2] = old_to_new[f[2]];
    }

    // Rebuild edge structures
    m.edges().clear();
    m.edge_ind_map().clear();
    m.adjacency().clear();
    m.incident_edges().clear();
    for (const auto& [v0, v1, v2] : m.faces()) {
        std::array<std::pair<index_t, index_t>, 3> face_edges = {{
            {std::min(v0, v1), std::max(v0, v1)},
            {std::min(v1, v2), std::max(v1, v2)},
            {std::min(v2, v0), std::max(v2, v0)}
        }};

        for (const auto& [eu, ev] : face_edges) {
            auto key = std::make_pair(eu, ev);
            if (!m.edge_ind_map().count(key)) {
                index_t eidx = m.edges().size();
                m.edges().push_back({eu, ev});
                m.edge_ind_map()[key] = eidx;
                m.adjacency()[eu].push_back(ev);
                m.adjacency()[ev].push_back(eu);
                m.incident_edges()[eu].push_back(eidx);
                m.incident_edges()[ev].push_back(eidx);
            }
        }
    }

    m.validate();
}

} // namespace mesh
} // namespace gnnmath
