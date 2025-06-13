#include "../include/gnnmath/mesh.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <execution>
#include <cmath>
#include <set>
#include <queue>

namespace gnnmath {
namespace mesh {

/**
 * @brief Loads a triangular mesh from an OBJ file.
 * @param filename Path to the OBJ file.
 * @throws std::runtime_error If the file cannot be opened or contains invalid data.
 */
void mesh::load_obj(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("load_obj: cannot open file " + filename);
    }

    vertices_.clear();
    edges_.clear();
    faces_.clear();
    adjacency_.clear();
    incident_edges_.clear();
    edge_index_map_.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        if (prefix == "v") {
            double x, y, z;
            if (!(iss >> x >> y >> z)) {
                throw std::runtime_error("load_obj: invalid vertex format");
            }
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                throw std::runtime_error("load_obj: non-finite vertex coordinate");
            }
        
            vertices_.push_back({x, y, z});
        } 
        else if (prefix == "f") {
            std::size_t v0, v1, v2;
            if (!(iss >> v0 >> v1 >> v2)) {
                throw std::runtime_error("load_obj: invalid face format");
            }
            if (v0 == 0 || v1 == 0 || v2 == 0 || 
                v0 > vertices_.size() || v1 > vertices_.size() || v2 > vertices_.size()) {
                throw std::runtime_error("load_obj: invalid vertex index");
            }
        
            faces_.push_back({v0 - 1, v1 - 1, v2 - 1});
        }
    }

    if (vertices_.empty()) {
        throw std::runtime_error("load_obj: no vertices found");
    }

    for (const auto& [v0, v1, v2] : faces_) {
        std::vector<std::pair<std::size_t, std::size_t>> face_edges = {
            {std::min(v0, v1), std::max(v0, v1)},
            {std::min(v1, v2), std::max(v1, v2)},
            {std::min(v2, v0), std::max(v2, v0)}
        };
        
        for (const auto& [u, v] : face_edges) {
            if (!edge_index_map_.count({u, v})) {
                std::size_t edge_idx = edges_.size();
                edges_.push_back({u, v});
                edge_index_map_[{u, v}] = edge_idx;
                adjacency_[u].push_back(v);
                adjacency_[v].push_back(u);
                incident_edges_[u].push_back(edge_idx);
                incident_edges_[v].push_back(edge_idx);
            }
        }
    }

    for (auto& [v, neighbors] : adjacency_) {
        if (neighbors.size() > 100) {
            std::sort(std::execution::par_unseq, neighbors.begin(), neighbors.end());
        } 
        else {
            std::sort(neighbors.begin(), neighbors.end());
        }
        
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    
    for (auto& [v, edges] : incident_edges_) {
        if (edges.size() > 100) {
            std::sort(std::execution::par_unseq, edges.begin(), edges.end());
        } 
        else {
            std::sort(edges.begin(), edges.end());
        }
        
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    }

    validate();
}

/// @brief Checks if the mesh is valid without throwing.
/// @return True if the mesh is valid, false otherwise.
bool mesh::is_valid() const {
    if (vertices_.empty()) {
        return false;
    }
    
    for (const auto& v : vertices_) {
        if (v.size() != 3) {
            return false;
        }
        
        for (double x : v) {
            if (!std::isfinite(x)) 
                return false;
        }
    }
    
    for (const auto& f : faces_) {
        for (std::size_t idx : f) {
            if (idx >= vertices_.size()) return false;
        }
        
        if (f[0] == f[1] || f[1] == f[2] || f[2] == f[0]) {
            return false;
        }
    }
    
    for (const auto& e : edges_) {
        if (e.first >= vertices_.size() || e.second >= vertices_.size() || e.first == e.second) 
            return false;
    }
    
    for (const auto& [v, neighbors] : adjacency_) {
        if (v >= vertices_.size()) 
            return false;
        for (auto u : neighbors) {
            if (u >= vertices_.size()) 
                return false;
        }
    }
    
    return true;
}

/**
 * @brief Returns the neighboring vertices of a given vertex.
 * @param vertex_idx Index of the vertex.
 * @return List of neighbor vertex indices.
 * @throws std::runtime_error If vertex_idx is invalid.
 */
std::vector<std::size_t> mesh::get_neighbors(std::size_t vertex_idx) const {
    if (vertex_idx >= vertices_.size()) {
        throw std::runtime_error("get_neighbors: invalid vertex index");
    }
    
    auto it = adjacency_.find(vertex_idx);
    return it != adjacency_.end() ? it->second : std::vector<std::size_t>{};
}

/**
 * @brief Returns the indices of edges incident to a given vertex.
 * @param vertex_idx Index of the vertex.
 * @return List of incident edge indices.
 * @throws std::runtime_error If vertex_idx is invalid.
 */
std::vector<std::size_t> mesh::get_incident_edges(std::size_t vertex_idx) const {
    if (vertex_idx >= vertices_.size()) {
        throw std::runtime_error("get_incident_edges: invalid vertex index");
    }

    auto it = incident_edges_.find(vertex_idx);
    return it != incident_edges_.end() ? it->second : std::vector<std::size_t>{};
}

/**
 * @brief Computes node features for GNN input (vertex coordinates).
 * @return Vector of node feature vectors.
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<gnnmath::vector::vector> mesh::compute_node_features() const {
    validate();
    return vertices_;
}

/**
 * @brief Computes edge features for GNN input (edge lengths).
 * @return Vector of edge feature vectors.
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<gnnmath::vector::vector> mesh::compute_edge_features() const {
    validate();
    std::vector<gnnmath::vector::vector> edge_features(edges_.size());
    std::transform(std::execution::par_unseq, edges_.begin(), edges_.end(), edge_features.begin(),
                   [this](const auto& edge) {
                       const auto& [u, v] = edge;
                       const auto& p0 = vertices_[u];
                       const auto& p1 = vertices_[v];
                       double len = gnnmath::vector::euclidean_norm(
                           gnnmath::vector::operator-(p1, p0));
    
                        return gnnmath::vector::vector{len};
                   });
    
    return edge_features;
}

/**
 * @brief Converts the mesh to a sparse adjacency matrix for GNN input.
 * @return Sparse adjacency matrix.
 * @throws std::runtime_error If the mesh is invalid.
 */
matrix::sparse_matrix mesh::to_adjacency_matrix() const {
    validate();
    return matrix::build_adj_matrix(n_vertices(), edges_);
}

/**
 * @brief Computes per-vertex normals as node features.
 * @return Vector of normal vectors.
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<gnnmath::vector::vector> mesh::compute_normals() const {
    validate();
    std::vector<gnnmath::vector::vector> normals(n_vertices(), {0.0, 0.0, 0.0});
    for (const auto& [v0, v1, v2] : faces_) {
        const auto& p0 = vertices_[v0];
        const auto& p1 = vertices_[v1];
        const auto& p2 = vertices_[v2];
        auto e1 = gnnmath::vector::operator-(p1, p0);
        auto e2 = gnnmath::vector::operator-(p2, p0);
        gnnmath::vector::vector normal = {
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        };
    
        double norm = gnnmath::vector::euclidean_norm(normal);
        if (norm > 1e-10) {
            normal[0] /= norm;
            normal[1] /= norm;
            normal[2] /= norm;
        }
    
        normals[v0] = gnnmath::vector::operator+(normals[v0], normal);
        normals[v1] = gnnmath::vector::operator+(normals[v1], normal);
        normals[v2] = gnnmath::vector::operator+(normals[v2], normal);
    }
    
    std::transform(std::execution::par_unseq, normals.begin(), normals.end(), normals.begin(),
                   [](auto& n) {
                       double norm = gnnmath::vector::euclidean_norm(n);
                       if (norm > 1e-10) {
                           n[0] /= norm;
                           n[1] /= norm;
                           n[2] /= norm;
                       }
    
                       return n;
                   });

    return normals;
}

/**
 * @brief Randomly samples a specified number of vertices.
 * @param n Number of vertices to sample.
 * @return List of sampled vertex indices.
 * @throws std::invalid_argument If n exceeds the vertex count.
 * @throws std::runtime_error If the mesh is invalid.
 */
std::vector<std::size_t> mesh::sample_vertices(std::size_t n) const {
    validate();
    if (n > n_vertices()) {
        throw std::invalid_argument("sample_vertices: n exceeds vertex count");
    }

    std::vector<std::size_t> indices(n_vertices());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<std::size_t> sample;
    sample.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t idx = static_cast<std::size_t>(
            random::uniform(0, static_cast<scalar_t>(indices.size() - 1)));
        sample.push_back(indices[idx]);
        indices[idx] = indices.back();
        indices.pop_back();
    }

    return sample;
}

/**
 * @brief Adds random noise to vertex coordinates.
 * @param scale Noise scale (e.g., 0.01 for small perturbations).
 * @throws std::invalid_argument If scale is negative.
 * @throws std::runtime_error If the mesh is invalid or results in non-finite coordinates.
 */
void mesh::add_vertex_noise(scalar_t scale) {
    validate();
    if (scale < 0) {
        throw std::invalid_argument("add_vertex_noise: scale must be non-negative");
    }

    random::seed(42);
    for (auto& v : vertices_) {
        auto noise = random::uniform_vector(3, -scale, scale);
        v = gnnmath::vector::operator+(v, noise);
        for (double& coord : v) {
            if (!std::isfinite(coord)) {
                throw std::runtime_error("add_vertex_noise: non-finite coordinate");
            }
        }
    }

    validate();
}

/**
 * @brief Simplifies the mesh by edge collapse to a target vertex count.
 * @param target_vertices Desired number of vertices.
 * @throws std::invalid_argument If target_vertices exceeds the current vertex count.
 * @throws std::runtime_error If the mesh is invalid.
 */
void mesh::simplify_edge_collapse(std::size_t target_vertices) {
    validate();
    if (target_vertices > n_vertices()) {
        throw std::invalid_argument("simplify_edge_collapse: target exceeds current vertices");
    }

    using cost_t = std::pair<double, std::size_t>;
    std::priority_queue<cost_t, std::vector<cost_t>, std::greater<cost_t>> pq;
    for (std::size_t i = 0; i < edges_.size(); ++i) {
        const auto& [u, v] = edges_[i];
        double cost = gnnmath::vector::euclidean_norm(
            gnnmath::vector::operator-(vertices_[u], vertices_[v]));
        pq.push({cost, i});
    }

    std::vector<bool> valid_vertices(n_vertices(), true);
    std::vector<bool> valid_edges(n_edges(), true);
    std::size_t current_vertices = n_vertices();
    while (current_vertices > target_vertices && !pq.empty()) {
        auto [cost, edge_idx] = pq.top();
        pq.pop();
        if (!valid_edges[edge_idx]) {
            continue;
        }

        auto [u, v] = edges_[edge_idx];
        if (!valid_vertices[u] || !valid_vertices[v]) {
            continue;
        }

        valid_vertices[u] = false;
        --current_vertices;
        valid_edges[edge_idx] = false;
        vertices_[v] = gnnmath::vector::scalar_multiply(
            gnnmath::vector::operator+(vertices_[u], vertices_[v]), 0.5);
        std::vector<face> new_faces;
        for (const auto& f : faces_) {
            face new_f = f;
            if (f[0] == u) new_f[0] = v;
            if (f[1] == u) new_f[1] = v;
            if (f[2] == u) new_f[2] = v;
            if (new_f[0] != new_f[1] && new_f[1] != new_f[2] && new_f[2] != new_f[0]) {
                new_faces.push_back(new_f);
            }
        }

        faces_ = std::move(new_faces);
        edges_.clear();
        edge_index_map_.clear();
        adjacency_.clear();
        incident_edges_.clear();
        for (const auto& [v0, v1, v2] : faces_) {
            std::vector<std::pair<std::size_t, std::size_t>> face_edges = {
                {std::min(v0, v1), std::max(v0, v1)},
                {std::min(v1, v2), std::max(v1, v2)},
                {std::min(v2, v0), std::max(v2, v0)}
            };

            for (const auto& [u2, v2] : face_edges) {
                if (!edge_index_map_.count({u2, v2})) {
                    std::size_t edge_idx = edges_.size();
                    edges_.push_back({u2, v2});
                    edge_index_map_[{u2, v2}] = edge_idx;
                    adjacency_[u2].push_back(v2);
                    adjacency_[v2].push_back(u2);
                    incident_edges_[u2].push_back(edge_idx);
                    incident_edges_[v2].push_back(edge_idx);
                }
            }
        }

        while (!pq.empty()) pq.pop();
        for (std::size_t i = 0; i < edges_.size(); ++i) {
            if (!valid_edges[i]) continue;
            const auto& [u2, v2] = edges_[i];
            if (!valid_vertices[u2] || !valid_vertices[v2]) {
                valid_edges[i] = false;
                continue;
            }

            double cost = gnnmath::vector::euclidean_norm(
                gnnmath::vector::operator-(vertices_[u2], vertices_[v2]));
            pq.push({cost, i});
        }
    }

    std::vector<vertex> new_vertices;
    std::vector<std::size_t> old_to_new(n_vertices(), 0);
    std::size_t new_idx = 0;
    for (std::size_t i = 0; i < n_vertices(); ++i) {
        if (valid_vertices[i]) {
            new_vertices.push_back(vertices_[i]);
            old_to_new[i] = new_idx++;
        }
    }

    vertices_ = std::move(new_vertices);
    for (auto& f : faces_) {
        f[0] = old_to_new[f[0]];
        f[1] = old_to_new[f[1]];
        f[2] = old_to_new[f[2]];
    }

    edges_.clear();
    edge_index_map_.clear();
    adjacency_.clear();
    incident_edges_.clear();
    for (const auto& [v0, v1, v2] : faces_) {
        std::vector<std::pair<std::size_t, std::size_t>> face_edges = {
            {std::min(v0, v1), std::max(v0, v1)},
            {std::min(v1, v2), std::max(v1, v2)},
            {std::min(v2, v0), std::max(v2, v0)}
        };

        for (const auto& [u2, v2] : face_edges) {
            if (!edge_index_map_.count({u2, v2})) {
                std::size_t edge_idx = edges_.size();
                edges_.push_back({u2, v2});
                edge_index_map_[{u2, v2}] = edge_idx;
                adjacency_[u2].push_back(v2);
                adjacency_[v2].push_back(u2);
                incident_edges_[u2].push_back(edge_idx);
                incident_edges_[v2].push_back(edge_idx);
            }
        }
    }

    validate();
}

/**
 * @brief Validates the mesh for consistency and correctness.
 * @throws std::runtime_error If the mesh is invalid (e.g., empty vertices, invalid indices).
 */
void mesh::validate() const {
    if (vertices_.empty()) {
        throw std::runtime_error("validate: no vertices");
    }

    for (const auto& [u, v] : edges_) {
        if (u >= vertices_.size() || v >= vertices_.size()) {
            throw std::runtime_error("validate: invalid edge vertex index");
        }
    }

    for (const auto& [v0, v1, v2] : faces_) {
        if (v0 >= vertices_.size() || v1 >= vertices_.size() || v2 >= vertices_.size()) {
            throw std::runtime_error("validate: invalid face vertex index");
        }
        if (v0 == v1 || v1 == v2 || v2 == v0) {
            throw std::runtime_error("validate: degenerate face");
        }
    }

    for (const auto& vertex : vertices_) {
        if (vertex.size() != 3) {
            throw std::runtime_error("validate: invalid vertex dimension");
        }

        for (double coord : vertex) {
            if (!std::isfinite(coord)) {
                throw std::runtime_error("validate: non-finite vertex coordinate");
            }
        }
    }

    for (const auto& [v, neighbors] : adjacency_) {
        if (v >= vertices_.size()) {
            throw std::runtime_error("validate: invalid adjacency vertex");
        }

        for (auto u : neighbors) {
            if (u >= vertices_.size()) {
                throw std::runtime_error("validate: invalid neighbor index");
            }
        }
    }
}

} // namespace mesh
} // namespace gnnmath
