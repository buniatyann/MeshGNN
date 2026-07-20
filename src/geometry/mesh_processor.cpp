#include <gnnmath/geometry/mesh.hpp>
#include <gnnmath/math/vector.hpp>
#include <gnnmath/math/dense_matrix.hpp>
#include <gnnmath/math/sparse_matrix.hpp>
#include <gnnmath/core/random.hpp>
#include <vector>
#include <queue>
#include <set>
#include <array>
#include <stdexcept>
#include <cmath>
#include <execution>
#include <functional>

namespace gnnmath {
namespace mesh {

namespace {

/// Symmetric 4x4 error quadric (Garland-Heckbert), stored as the upper
/// triangle: q00 q01 q02 q03 q11 q12 q13 q22 q23 q33.
struct quadric {
    std::array<scalar_t, 10> q{};

    quadric& operator+=(const quadric& rhs) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            q[i] += rhs.q[i];
        }

        return *this;
    }

    quadric operator+(const quadric& rhs) const {
        quadric result = *this;
        result += rhs;
        return result;
    }

    /// v^T Q v with v = (x, y, z, 1); clamped at 0 against rounding noise.
    scalar_t evaluate(const mesh::vertex& p) const {
        scalar_t x = p[0], y = p[1], z = p[2];
        scalar_t val = q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x
                     + q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y
                     + q[7] * z * z + 2.0 * q[8] * z
                     + q[9];
        return std::max(0.0, val);
    }
};

/// Plane quadric of the triangle (p0, p1, p2): K = pp^T for the unit plane
/// p = (a, b, c, d) with ax + by + cz + d = 0. Degenerate faces contribute zero.
quadric plane_quadric(const mesh::vertex& p0, const mesh::vertex& p1, const mesh::vertex& p2) {
    scalar_t e1x = p1[0] - p0[0], e1y = p1[1] - p0[1], e1z = p1[2] - p0[2];
    scalar_t e2x = p2[0] - p0[0], e2y = p2[1] - p0[1], e2z = p2[2] - p0[2];
    scalar_t a = e1y * e2z - e1z * e2y;
    scalar_t b = e1z * e2x - e1x * e2z;
    scalar_t c = e1x * e2y - e1y * e2x;
    scalar_t len = std::sqrt(a * a + b * b + c * c);

    quadric k;
    if (len < 1e-12) {
        return k;
    }

    a /= len;
    b /= len;
    c /= len;
    scalar_t d = -(a * p0[0] + b * p0[1] + c * p0[2]);
    k.q = {a * a, a * b, a * c, a * d,
           b * b, b * c, b * d,
           c * c, c * d,
           d * d};
    return k;
}

/// Per-vertex quadrics: each face's plane quadric accumulates into its vertices.
std::vector<quadric> compute_vertex_quadrics(const mesh& m) {
    std::vector<quadric> quadrics(m.n_vertices());
    for (const auto& [v0, v1, v2] : m.faces()) {
        quadric k = plane_quadric(m.vertices()[v0], m.vertices()[v1], m.vertices()[v2]);
        quadrics[v0] += k;
        quadrics[v1] += k;
        quadrics[v2] += k;
    }

    return quadrics;
}

mesh::vertex midpoint(const mesh::vertex& a, const mesh::vertex& b) {
    return {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])};
}

} // namespace

scalar_t compute_quadric_error(const mesh& m, index_t u, index_t v) {
    if (u >= m.n_vertices() || v >= m.n_vertices()) {
        throw std::runtime_error("compute_quadric_error: invalid vertex index");
    }

    // Garland-Heckbert cost of collapsing (u, v) to the edge midpoint:
    // (Qu + Qv) evaluated at the midpoint, with Qu/Qv accumulated from the
    // planes of all faces incident to u or v.
    quadric qu, qv;
    for (const auto& f : m.faces()) {
        bool has_u = (f[0] == u || f[1] == u || f[2] == u);
        bool has_v = (f[0] == v || f[1] == v || f[2] == v);
        if (!has_u && !has_v) {
            continue;
        }

        quadric k = plane_quadric(m.vertices()[f[0]], m.vertices()[f[1]], m.vertices()[f[2]]);
        if (has_u) {
            qu += k;
        }
        if (has_v) {
            qv += k;
        }
    }

    return (qu + qv).evaluate(midpoint(m.vertices()[u], m.vertices()[v]));
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

    // collapse_target[u] = v means u was merged into v; chains are possible
    // (u -> v -> w) when the survivor of one collapse is later removed itself.
    std::vector<index_t> collapse_target(m.n_vertices());
    for (index_t i = 0; i < m.n_vertices(); ++i) {
        collapse_target[i] = i;
    }

    // Per-vertex error quadrics (Garland-Heckbert), accumulated on collapse
    std::vector<quadric> quadrics = compute_vertex_quadrics(m);
    auto edge_cost = [&](index_t eu, index_t ev) {
        return (quadrics[eu] + quadrics[ev]).evaluate(midpoint(m.vertices()[eu], m.vertices()[ev]));
    };

    // costs initialization
    const auto& edges = m.edges();
    if (!gnn_scores.empty()) {
        costs = gnn_scores;
    }
    else {
        for (index_t i = 0; i < edges.size(); ++i) {
            costs[i] = edge_cost(edges[i].first, edges[i].second);
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
        collapse_target[u] = v;
        --current_vertices;

        // Update vertex position to midpoint; the survivor inherits the
        // removed vertex's accumulated quadric
        m.vertices()[v] = vector::scalar_multiply(
            vector::operator+(m.vertices()[u], m.vertices()[v]), 0.5);
        quadrics[v] += quadrics[u];

        // Collect edges that need cost updates (edges incident to v)
        std::set<index_t> edges_to_update;

        // Current neighbors of v (to detect duplicate edges after redirection)
        std::set<index_t> v_neighbors;
        if (m.incident_edges().count(v)) {
            for (index_t inc_edge : m.incident_edges().at(v)) {
                if (valid_edges[inc_edge]) {
                    auto [eu, ev] = m.edges()[inc_edge];
                    v_neighbors.insert(eu == v ? ev : eu);
                    edges_to_update.insert(inc_edge);
                }
            }
        }

        // Redirect u's incident edges to v and transfer them to v's incidence list
        if (m.incident_edges().count(u)) {
            for (index_t inc_edge : m.incident_edges().at(u)) {
                if (valid_edges[inc_edge] && inc_edge != edge_idx) {
                    auto& [eu, ev] = m.edges()[inc_edge];
                    index_t other = (eu == u) ? ev : eu;
                    // Self-loop or duplicate of an existing v-edge
                    if (other == v || v_neighbors.count(other)) {
                        valid_edges[inc_edge] = false;
                        continue;
                    }

                    if (eu == u) {
                        eu = v;
                    }
                    if (ev == u) {
                        ev = v;
                    }

                    m.incident_edges()[v].push_back(inc_edge);
                    v_neighbors.insert(other);
                    edges_to_update.insert(inc_edge);
                }
            }

            m.incident_edges().erase(u);
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

            // New cost from the updated quadrics. Note: even when the initial
            // costs came from gnn_scores, redirected/moved edges are re-costed
            // geometrically (a GNN score for the old geometry is stale).
            costs[eidx] = edge_cost(eu, ev);
            edge_versions[eidx]++;
            pq.push({costs[eidx], eidx, edge_versions[eidx]});
        }
    }

    // Resolve collapse chains (u -> v -> w) to the final surviving vertex,
    // with path compression
    std::function<index_t(index_t)> resolve = [&](index_t i) -> index_t {
        if (collapse_target[i] != i) {
            collapse_target[i] = resolve(collapse_target[i]);
        }

        return collapse_target[i];
    };

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

    // remap faces onto surviving vertices and filter degenerate faces
    std::vector<mesh::face> new_faces;
    for (const auto& f : m.faces()) {
        mesh::face new_f = {old_to_new[resolve(f[0])], old_to_new[resolve(f[1])], old_to_new[resolve(f[2])]};
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
