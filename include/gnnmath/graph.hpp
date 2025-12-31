#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "math/vector.hpp"
#include "math/dense_matrix.hpp"
#include "math/sparse_matrix.hpp"
#include "geometry/mesh.hpp"
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <utility>
#include <string>

namespace gnnmath {
namespace graph {

using sparse_matrix = gnnmath::matrix::sparse_matrix; ///< Sparse matrix type.
using feature_vector = std::vector<gnnmath::vector::vector>; ///< Vector of feature vectors.

/**
 * @brief Represents an undirected graph with node and edge features for GNN processing.
 */
struct graph {
    using vector = gnnmath::vector::vector; ///< Vector for features.
    using edge = std::pair<std::size_t, std::size_t>; ///< Edge as vertex indices (u, v).
    using vector_container = std::vector<vector>; ///< Container for vectors.
    using edge_container = std::vector<edge>; ///< Container for edges.
    using adjacency_map = std::unordered_map<std::size_t, std::vector<std::pair<std::size_t, std::size_t>>>; ///< Adjacency map type.

    std::size_t num_vertices; ///< Number of vertices in the graph.
    edge_container edges; ///< List of undirected edges (u, v).
    vector_container node_features; ///< Feature vectors for each vertex.
    vector_container edge_features; ///< Feature vectors for each edge.
    adjacency_map adjacency; ///< Adjacency list: vertex -> (neighbor, edge_index).

    /**
     * @brief Constructs a graph with specified vertices, edges, and features.
     * @param num_vertices Number of vertices.
     * @param edges List of edge pairs (u, v).
     * @param node_features Feature vectors for each vertex.
     * @param edge_features Feature vectors for each edge.
     * @throws std::runtime_error If inputs are invalid (e.g., size mismatch, invalid vertices).
     */
    graph(std::size_t num_vertices,
          const edge_container& edges,
          const vector_container& node_features,
          const vector_container& edge_features);
};

/**
 * @brief Validates the graph's integrity.
 * @param graph The graph to validate.
 * @throws std::runtime_error If the graph is invalid (e.g., invalid vertices, inconsistent features).
 */
void validate(const graph& graph);

/**
 * @brief Constructs a graph from a mesh object.
 * @param mesh The input mesh.
 * @param node_features Feature vectors for each vertex.
 * @param edge_features Feature vectors for each edge.
 * @return A graph object representing the mesh.
 * @throws std::runtime_error If inputs are invalid (e.g., size mismatch, inconsistent features).
 */
graph from_mesh(const mesh::mesh& mesh,
                const feature_vector& node_features,
                const feature_vector& edge_features);

/**
 * @brief Converts the graph to a sparse adjacency matrix in CSR format.
 * @param graph The input graph.
 * @return A sparse_matrix representing the adjacency matrix.
 * @throws std::runtime_error If the graph is invalid.
 */
sparse_matrix to_adjacency_matrix(const graph& graph);

/**
 * @brief Performs one step of message passing, updating node features.
 * @param graph The input graph.
 * @param messages Message vectors for each edge.
 * @param edge_weights Weights for each edge.
 * @return Updated node feature vectors.
 * @throws std::runtime_error If inputs are invalid (e.g., size mismatch, inconsistent dimensions).
 */
feature_vector message_passing(
    const graph& graph,
    const feature_vector& messages,
    const gnnmath::vector::vector& edge_weights);

/**
 * @brief Aggregates neighbor features for each node.
 * @param graph The input graph.
 * @param node_features Feature vectors for each vertex.
 * @param mode Aggregation mode ("sum", "mean", "max").
 * @return Aggregated feature vectors for each vertex.
 * @throws std::runtime_error If inputs are invalid (e.g., invalid mode, size mismatch).
 */
feature_vector aggregate_features(
    const graph& graph,
    const feature_vector& node_features,
    const std::string& mode);

/**
 * @brief Updates the graph's node and edge features.
 * @param graph The graph to modify.
 * @param node_features New feature vectors for each vertex.
 * @param edge_features New feature vectors for each edge.
 * @throws std::runtime_error If inputs are invalid (e.g., size mismatch, inconsistent dimensions).
 */
void update_features(graph& graph,
                     const feature_vector& node_features,
                     const feature_vector& edge_features);

/**
 * @brief Retrieves the neighboring vertices of a given vertex.
 * @param graph The input graph.
 * @param vertex Vertex index.
 * @return List of neighbor vertex indices.
 * @throws std::runtime_error If vertex is invalid.
 */
std::vector<std::size_t> get_neighbors(const graph& graph, std::size_t vertex);

/**
 * @brief Computes the degree of each vertex.
 * @param graph The input graph.
 * @return Vector of degrees for each vertex.
 * @throws std::runtime_error If the graph is invalid.
 */
std::vector<std::size_t> compute_degree(const graph& graph);

/**
 * @brief Constructs the graph Laplacian matrix in CSR format.
 * @param graph The input graph.
 * @return A sparse_matrix representing the Laplacian.
 * @throws std::runtime_error If the graph is invalid.
 */
sparse_matrix laplacian_matrix(const graph& graph);

} // namespace graph
} // namespace gnnmath

#endif 