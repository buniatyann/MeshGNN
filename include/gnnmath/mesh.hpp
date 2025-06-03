#pragma once

#include "vector.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <utility>

namespace gnnmath {
namespace mesh {

/**
 * @brief Represents a triangular mesh with vertices, edges, and faces.
 */
class mesh {
public:
    using vertex = gnnmath::vector::vector; ///< Vertex coordinates (x, y, z).
    using edge = std::pair<std::size_t, std::size_t>; ///< Edge as vertex indices (u, v).
    using face = std::array<std::size_t, 3>; ///< Face as three vertex indices.

    /**
     * @brief Constructs an empty mesh.
     */
    mesh() = default;

    /**
     * @brief Loads a mesh from an OBJ file.
     * @param filename Path to the OBJ file.
     * @throws std::runtime_error If file is invalid or cannot be read.
     */
    void load_obj(const std::string& filename);

    /**
     * @brief Returns the number of vertices.
     * @return Number of vertices.
     */
    std::size_t n_vertices() const { return vertices_.size(); }

    /**
     * @brief Returns the number of edges.
     * @return Number of edges.
     */
    std::size_t n_edges() const { return edges_.size(); }

    /**
     * @brief Returns the number of faces.
     * @return Number of faces.
     */
    std::size_t n_faces() const { return faces_.size(); }

    /**
     * @brief Returns the list of vertices.
     * @return Vector of vertex coordinates.
     */
    const std::vector<vertex>& vertices() const { return vertices_; }

    /**
     * @brief Returns the list of edges.
     * @return Vector of edge pairs.
     */
    const std::vector<edge>& edges() const { return edges_; }

    /**
     * @brief Returns the list of faces.
     * @return Vector of face triplets.
     */
    const std::vector<face>& faces() const { return faces_; }

    /**
     * @brief Returns the neighbors of a vertex.
     * @param vertex_idx Vertex index.
     * @return List of neighboring vertex indices.
     * @throws std::runtime_error If vertex_idx is invalid.
     */
    std::vector<std::size_t> get_neighbors(std::size_t vertex_idx) const;

    /**
     * @brief Returns the edges incident to a vertex.
     * @param vertex_idx Vertex index.
     * @return List of edge indices.
     * @throws std::runtime_error If vertex_idx is invalid.
     */
    std::vector<std::size_t> get_incident_edges(std::size_t vertex_idx) const;

    /**
     * @brief Computes node features (vertex coordinates).
     * @return Vector of vertex coordinate features.
     */
    std::vector<gnnmath::vector::vector> compute_node_features() const;

    /**
     * @brief Computes edge features (edge lengths).
     * @return Vector of edge length features.
     */
    std::vector<gnnmath::vector::vector> compute_edge_features() const;

    /**
     * @brief Validates the mesh structure.
     * @throws std::runtime_error If the mesh is invalid (e.g., invalid indices).
     */
    void validate() const;

private:
    std::vector<vertex> vertices_; // Vertex coordinates (x, y, z)
    std::vector<edge> edges_; // Edges as (u, v) pairs
    std::vector<face> faces_; // Faces as (v0, v1, v2) triplets
    std::unordered_map<std::size_t, std::vector<std::size_t>> adjacency_; // Vertex -> neighbors
    std::unordered_map<std::size_t, std::vector<std::size_t>> incident_edges_; // Vertex -> edge indices

    // Helper for edge indexing
    struct pair_hash {
        std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const {
            return std::hash<std::size_t>{}(p.first) ^ std::hash<std::size_t>{}(p.second);
        }
    };
    
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, pair_hash> edge_index_map_;
};

} // namespace mesh
} // namespace gnnmath