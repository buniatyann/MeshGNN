#include <gnnmath/geometry/mesh.hpp>
#include <gnnmath/geometry/obj_loader.hpp>
#include <gnnmath/geometry/mesh_processor.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <execution>
#include <cmath>
#include <numeric>

namespace gnnmath {
namespace mesh {

void mesh::load_obj(const std::string& filename) {
    obj_load_options options;
    options.triangulate = true;
    options.generate_normals = false;
    load_obj(filename, options);
}

void mesh::load_obj(const std::string& filename, const obj_load_options& options) {
    obj_loader loader(options);
    obj_data data = loader.load(filename);
    load_from_obj_data(data);
}

void mesh::load_from_obj_data(const obj_data& data) {
    vertices_.clear();
    edges_.clear();
    faces_.clear();
    texcoords_.clear();
    file_normals_.clear();
    faces_with_attrs_.clear();
    adjacency_.clear();
    incident_edges_.clear();
    edge_index_map_.clear();

    vertices_.reserve(data.vertices.size());
    for (const auto& v : data.vertices) {
        vertices_.push_back({v[0], v[1], v[2]});
    }

    // texture coordinates
    texcoords_.reserve(data.texcoords.size());
    for (const auto& vt : data.texcoords) {
        texcoords_.push_back({vt[0], vt[1]});
    }

    // normals
    file_normals_.reserve(data.normals.size());
    for (const auto& vn : data.normals) {
        file_normals_.push_back({vn[0], vn[1], vn[2]});
    }

    // Convert faces
    faces_.reserve(data.faces.size());
    faces_with_attrs_.reserve(data.faces.size());
    for (const auto& poly_face : data.faces) {
        if (poly_face.size() < 3) {
            continue;
        }

        // Simple face (vertex indices only)
        face simple_face = {
            poly_face[0].vertex_idx,
            poly_face[1].vertex_idx,
            poly_face[2].vertex_idx
        };
        
        faces_.push_back(simple_face);

        // Face with full attributes
        face_with_attrs full_face;
        full_face.vertex_indices = simple_face;
        for (std::size_t i = 0; i < 3; ++i) {
            full_face.texcoord_indices[i] = poly_face[i].texcoord_idx;
            full_face.normal_indices[i] = poly_face[i].normal_idx;
        }
        
        faces_with_attrs_.push_back(full_face);
    }

    if (vertices_.empty()) {
        throw std::runtime_error("load_obj: no vertices found");
    }

    // Build edge and adjacency data structures
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

    // Sort and deduplicate adjacency lists (vertex degrees are small; a
    // parallel sort would be pure dispatch overhead)
    for (auto& [v, neighbors] : adjacency_) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    for (auto& [v, edge_list] : incident_edges_) {
        std::sort(edge_list.begin(), edge_list.end());
        edge_list.erase(std::unique(edge_list.begin(), edge_list.end()), edge_list.end());
    }

    validate();
}

bool mesh::is_valid() const {
    if (vertices_.empty()) {
        return false;
    }
    
    for (const auto& v : vertices_) {
        if (v.size() != 3) {
            return false;
        }
        
        for (double x : v) {
            if (!std::isfinite(x)) {
                return false;
            }
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
        if (e.first >= vertices_.size() || e.second >= vertices_.size() || e.first == e.second) {
            return false;
        }
    }
    
    for (const auto& [v, neighbors] : adjacency_) {
        if (v >= vertices_.size()) {
            return false;
        }
        
        for (auto u : neighbors) {
            if (u >= vertices_.size()) {
                return false;
            }
        }
    }
    
    return true;
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
    return vertices_;
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

matrix::sparse_matrix mesh::to_adjacency_matrix() const {
    validate();
    return matrix::build_adj_matrix(n_vertices(), edges_);
}

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
    
    for (auto& n : normals) {
        double norm = gnnmath::vector::euclidean_norm(n);
        if (norm > 1e-10) {
            n[0] /= norm;
            n[1] /= norm;
            n[2] /= norm;
        }
    }

    return normals;
}

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
        // Draw from [0, size) and truncate so every index is equally likely
        std::size_t idx = static_cast<std::size_t>(
            random::uniform(0, static_cast<scalar_t>(indices.size())));
        idx = std::min(idx, indices.size() - 1);
        sample.push_back(indices[idx]);
        indices[idx] = indices.back();
        indices.pop_back();
    }

    return sample;
}

void mesh::add_vertex_noise(scalar_t scale) {
    validate();
    if (scale < 0) {
        throw std::invalid_argument("add_vertex_noise: scale must be non-negative");
    }

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

void mesh::simplify_edge_collapse(std::size_t target_vertices) {
    // Delegates to the lazy-deletion implementation in mesh_processor
    // (O(E log E) instead of rebuilding all edges per collapse). Collapse
    // ordering may differ slightly from the old per-collapse-rebuild variant.
    simplify_gnn_edge_collapse(*this, target_vertices, {});
}

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
