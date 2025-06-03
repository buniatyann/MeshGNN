#include "../include/gnnmath/mesh.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <execution>
#include <cmath>

namespace gnnmath {
namespace mesh {

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
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        if (prefix == "v") {
            double x, y, z;
            if (!(iss >> x >> y >> z)) {
                throw std::runtime_error("load_obj: invalid vertex format");
            }
            
            vertices_.push_back({x, y, z});
        } 
        else if (prefix == "f") {
            std::size_t v0, v1, v2;
            if (!(iss >> v0 >> v1 >> v2)) {
                throw std::runtime_error("load_obj: invalid face format");
            }
            // OBJ indices are 1-based
            if (v0 == 0 || v1 == 0 || v2 == 0 || v0 > vertices_.size() || v1 > vertices_.size() || v2 > vertices_.size()) {
                throw std::runtime_error("load_obj: invalid vertex index");
            }
        
            faces_.push_back({v0 - 1, v1 - 1, v2 - 1});
        }
    }
    
    file.close();

    // Build edges and adjacency
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

    // sort adjacency lists for consistency
    for (auto& [v, neighbors] : adjacency_) {
        std::sort(std::execution::par_unseq, neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    
    for (auto& [v, edges] : incident_edges_) {
        std::sort(std::execution::par_unseq, edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    }

    validate();
}

std::vector<std::size_t> mesh::get_neighbors(std::size_t vertex_idx) const {
    if (vertex_idx >= vertices_.size()) {
        throw std::runtime_error("get_neighbors: invalid vertex index");
    }
    
    auto it = adjacency_.find(vertex_idx);
    return it != adjacency_.end() ? it->second : std::vector<std::size_t>{};
}

std::vector<std::size_t> mesh::get_incident_edges(std::size_t vertex_idx) const {
    if (vertex_idx >= vertices_.size()) {
        throw std::runtime_error("get_incident_edges: invalid vertex index");
    }
    
    auto it = incident_edges_.find(vertex_idx);
    return it != incident_edges_.end() ? it->second : std::vector<std::size_t>{};
}

std::vector<gnnmath::vector::vector> mesh::compute_node_features() const {
    validate();
    return vertices_; // Return vertex coordinates as features
}

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

void mesh::validate() const {
    for (const auto& [u, v] : edges_) {
        if (u >= vertices_.size() || v >= vertices_.size()) {
            throw std::runtime_error("validate: invalid edge vertex index");
        }
    }
    
    for (const auto& [v0, v1, v2] : faces_) {
        if (v0 >= vertices_.size() || v1 >= vertices_.size() || v2 >= vertices_.size()) {
            throw std::runtime_error("validate: invalid face vertex index");
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
}

} // namespace mesh
} // namespace gnnmath