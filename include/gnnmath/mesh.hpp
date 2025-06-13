#ifndef GNNMATH_MESH_HPP
#define GNNMATH_MESH_HPP

#include "vector.hpp"
#include "matrix.hpp"
#include "random.hpp"
#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <stdexcept>
#include <utility>

namespace gnnmath {
namespace mesh {

/**
 * @brief A class representing a triangular mesh for GNN-based simplification.
 *
 * The mesh class stores vertices, edges, and faces of a 3D triangular mesh,
 * providing functionality for loading OBJ files, computing features for GNNs,
 * and performing simplification operations.
 */
class mesh {
public:
    /// @brief Type alias for a vertex, represented as a vector of coordinates (x, y, z).
    using vertex = gnnmath::vector::vector;
    /// @brief Type alias for an edge, represented as a pair of vertex indices (u, v).
    using edge = std::pair<std::size_t, std::size_t>;
    /// @brief Type alias for a face, represented as three vertex indices forming a triangle.
    using face = std::array<std::size_t, 3>;

    /// @brief Default constructor, initializes an empty mesh.
    mesh() = default;

    /**
     * @brief Loads a triangular mesh from an OBJ file.
     * @param filename Path to the OBJ file.
     * @throws std::runtime_error If the file cannot be opened or contains invalid data.
     */
    void load_obj(const std::string& filename);

    /// @brief Checks if the mesh is valid without throwing.
    /// @return True if the mesh is valid, false otherwise.
    bool is_valid() const;

    /**
     * @brief Returns the number of vertices in the mesh.
     * @return Number of vertices.
     */
    std::size_t n_vertices() const { return vertices_.size(); }

    /**
     * @brief Returns the number of edges in the mesh.
     * @return Number of edges.
     */
    std::size_t n_edges() const { return edges_.size(); }

    /**
     * @brief Returns the number of faces in the mesh.
     * @return Number of faces.
     */
    std::size_t n_faces() const { return faces_.size(); }

    /**
     * @brief Returns the vertices of the mesh (const access).
     * @return Const reference to the vector of vertices.
     */
    const std::vector<vertex>& vertices() const { return vertices_; }

    /**
     * @brief Returns the vertices of the mesh (non-const access).
     * @return Reference to the vector of vertices.
     */
    std::vector<vertex>& vertices() { return vertices_; }

    /**
     * @brief Returns the edges of the mesh (const access).
     * @return Const reference to the vector of edges.
     */
    const std::vector<edge>& edges() const { return edges_; }

    /**
     * @brief Returns the edges of the mesh (non-const access).
     * @return Reference to the vector of edges.
     */
    std::vector<edge>& edges() { return edges_; }

    /**
     * @brief Returns the faces of the mesh (const access).
     * @return Const reference to the vector of faces.
     */
    const std::vector<face>& faces() const { return faces_; }

    /**
     * @brief Returns the faces of the mesh (non-const access).
     * @return Reference to the vector of faces.
     */
    std::vector<face>& faces() { return faces_; }

    /**
     * @brief Returns the adjacency map of the mesh (const access).
     * @return Const reference to the adjacency map (vertex index -> neighbor indices).
     */
    const std::unordered_map<std::size_t, std::vector<std::size_t>>& adjacency() const { return adjacency_; }

    /**
     * @brief Returns the adjacency map of the mesh (non-const access).
     * @return Reference to the adjacency map (vertex index -> neighbor indices).
     */
    std::unordered_map<std::size_t, std::vector<std::size_t>>& adjacency() { return adjacency_; }

private:
    /**
     * @brief Hash function for edge index map keys (vertex index pairs).
     */
    struct pair_hash {
        /**
         * @brief Computes hash for a pair of vertex indices.
         * @param p Pair of vertex indices (u, v).
         * @return Hash value.
         */
        std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const {
            auto h1 = std::hash<std::size_t>{}(p.first);
            auto h2 = std::hash<std::size_t>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

public:
    /**
     * @brief Returns the edge index map of the mesh (const access).
     * @return Const reference to the edge index map ((u, v) -> edge index).
     */
    const std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, pair_hash>& edge_ind_map() const { return edge_index_map_; }

    /**
     * @brief Returns the edge index map of the mesh (non-const access).
     * @return Reference to the edge index map ((u, v) -> edge index).
     */
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, pair_hash>& edge_ind_map() { return edge_index_map_; }

    /// @brief Accesses the vertices (mutable).
    /// @return Reference to the vertices vector.
    /// @throws std::runtime_error If the mesh is invalid.
    std::vector<vertex>& vertices() { return vertices_; }

    /// @brief Accesses the vertices (immutable).
    /// @return Const reference to the vertices vector.
    /// @throws std::runtime_error If the mesh is invalid.
    const std::vector<vertex>& vertices() const { return vertices_; }

    /**
     * @brief Returns the incident edges map of the mesh (const access).
     * @return Const reference to the incident edges map (vertex index -> edge indices).
     */
    const std::unordered_map<std::size_t, std::vector<std::size_t>>& incident_edges() const { return incident_edges_; }

    /**
     * @brief Returns the incident edges map of the mesh (non-const access).
     * @return Reference to the incident edges map (vertex index -> edge indices).
     */
    std::unordered_map<std::size_t, std::vector<std::size_t>>& incident_edges() { return incident_edges_; }

    /**
     * @brief Returns the neighboring vertices of a given vertex.
     * @param vertex_idx Index of the vertex.
     * @return List of neighbor vertex indices.
     * @throws std::runtime_error If vertex_idx is invalid.
     */
    std::vector<std::size_t> get_neighbors(std::size_t vertex_idx) const;

    /**
     * @brief Returns the indices of edges incident to a given vertex.
     * @param vertex_idx Index of the vertex.
     * @return List of incident edge indices.
     * @throws std::runtime_error If vertex_idx is invalid.
     */
    std::vector<std::size_t> get_incident_edges(std::size_t vertex_idx) const;

    /**
     * @brief Computes node features for GNN input (vertex coordinates).
     * @return Vector of node feature vectors.
     * @throws std::runtime_error If the mesh is invalid.
     */
    std::vector<gnnmath::vector::vector> compute_node_features() const;

    /**
     * @brief Computes edge features for GNN input (edge lengths).
     * @return Vector of edge feature vectors.
     * @throws std::runtime_error If the mesh is invalid.
     */
    std::vector<gnnmath::vector::vector> compute_edge_features() const;

    /**
     * @brief Converts the mesh to a sparse adjacency matrix for GNN input.
     * @return Sparse adjacency matrix.
     * @throws std::runtime_error If the mesh is invalid.
     */
    matrix::sparse_matrix to_adjacency_matrix() const;

    /**
     * @brief Computes per-vertex normals as node features.
     * @return Vector of normal vectors.
     * @throws std::runtime_error If the mesh is invalid.
     */
    std::vector<gnnmath::vector::vector> compute_normals() const;

    /**
     * @brief Randomly samples a specified number of vertices.
     * @param n Number of vertices to sample.
     * @return List of sampled vertex indices.
     * @throws std::invalid_argument If n exceeds the vertex count.
     * @throws std::runtime_error If the mesh is invalid.
     */
    std::vector<std::size_t> sample_vertices(std::size_t n) const;

    /**
     * @brief Adds random noise to vertex coordinates.
     * @param scale Noise scale (e.g., 0.01 for small perturbations).
     * @throws std::invalid_argument If scale is negative.
     * @throws std::runtime_error If the mesh is invalid or results in non-finite coordinates.
     */
    void add_vertex_noise(scalar_t scale);

    /**
     * @brief Simplifies the mesh by edge collapse to a target vertex count.
     * @param target_vertices Desired number of vertices.
     * @throws std::invalid_argument If target_vertices exceeds the current vertex count.
     * @throws std::runtime_error If the mesh is invalid.
     */
    void simplify_edge_collapse(std::size_t target_vertices);

    /**
     * @brief Validates the mesh for consistency and correctness.
     * @throws std::runtime_error If the mesh is invalid (e.g., empty vertices, invalid indices).
     */
    void validate() const;

private:
    /// @brief List of vertices, each storing 3D coordinates.
    std::vector<vertex> vertices_;
    /// @brief List of edges, each as a pair of vertex indices.
    std::vector<edge> edges_;
    /// @brief List of triangular faces, each as three vertex indices.
    std::vector<face> faces_;
    /// @brief Adjacency map: vertex index to list of neighbor vertex indices.
    std::unordered_map<std::size_t, std::vector<std::size_t>> adjacency_;
    /// @brief Incident edges map: vertex index to list of incident edge indices.
    std::unordered_map<std::size_t, std::vector<std::size_t>> incident_edges_;
    /// @brief Edge index map: (u, v) pair to edge index in edges_.
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, pair_hash> edge_index_map_;

    // Friend functions for direct access
    friend void simplify_gnn_edge_collapse(mesh&, index_t, const std::vector<scalar_t>&);
    friend void simplify_random_removal(mesh&, index_t);
};

} // namespace mesh
} // namespace gnnmath

#endif // GNNMATH_MESH_HPP