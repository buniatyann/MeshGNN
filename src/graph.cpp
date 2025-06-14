#include "../include/gnnmath/graph.hpp"
#include <algorithm>
#include <execution>
#include <numeric>

namespace gnnmath {
namespace graph {

using vector = gnnmath::vector::vector;
using edge = std::pair<std::size_t, std::size_t>;
using vector_container = std::vector<vector>;
using edge_container = std::vector<edge>;
using adjacency_map = std::unordered_map<std::size_t, std::vector<std::pair<std::size_t, std::size_t>>>;
using feature_vector = std::vector<gnnmath::vector::vector>;

graph::graph(std::size_t num_vertices,
             const edge_container& edges,
             const vector_container& node_features,
             const vector_container& edge_features)
    : num_vertices(num_vertices), edges(edges), node_features(node_features), edge_features(edge_features) {
    if (node_features.size() != num_vertices) {
        throw std::runtime_error("graph constructor: node_features size mismatch (" +
                                 std::to_string(node_features.size()) + " != " +
                                 std::to_string(num_vertices) + ")");
    }
    if (edge_features.size() != edges.size()) {
        throw std::runtime_error("graph constructor: edge_features size mismatch");
    }
    
    for (const auto& [u, v] : edges) {
        if (u >= num_vertices || v >= num_vertices) {
            throw std::runtime_error("graph constructor: invalid vertex index");
        }
    }
    
    if (!node_features.empty() && !std::all_of(node_features.begin(), node_features.end(),
        [&](const auto& f) { return f.size() == node_features[0].size(); })) {
        throw std::runtime_error("graph constructor: inconsistent node feature dimensions");
    }
    if (!edge_features.empty() && !std::all_of(edge_features.begin(), edge_features.end(),
        [&](const auto& f) { return f.size() == edge_features[0].size(); })) {
        throw std::runtime_error("graph constructor: inconsistent edge feature dimensions");
    }

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const auto& [u, v] = edges[i];
        adjacency[u].emplace_back(v, i);
        adjacency[v].emplace_back(u, i);
    }
}

void validate(const graph& graph) {
    if (graph.node_features.size() != graph.num_vertices) {
        throw std::runtime_error("validate: node_features size mismatch");
    }
    if (graph.edge_features.size() != graph.edges.size()) {
        throw std::runtime_error("validate: edge_features size mismatch");
    }
    
    for (const auto& [u, v] : graph.edges) {
        if (u >= graph.num_vertices || v >= graph.num_vertices) {
            throw std::runtime_error("validate: invalid vertex index");
        }
    }
    
    if (!graph.node_features.empty() && !std::all_of(graph.node_features.begin(), graph.node_features.end(),
        [&](const auto& f) { return f.size() == graph.node_features[0].size(); })) {
        throw std::runtime_error("validate: inconsistent node feature dimensions");
    }
    if (!graph.edge_features.empty() && !std::all_of(graph.edge_features.begin(), graph.edge_features.end(),
        [&](const auto& f) { return f.size() == graph.edge_features[0].size(); })) {
        throw std::runtime_error("validate: inconsistent edge feature dimensions");
    }
}

graph from_mesh(const mesh::mesh& mesh,
                const feature_vector& node_features,
                const feature_vector& edge_features) {
    if (node_features.size() != mesh.n_vertices()) {
        throw std::runtime_error("from_mesh: node_features size mismatch");
    }
    if (edge_features.size() != mesh.n_edges()) {
        throw std::runtime_error("from_mesh: edge_features size mismatch");
    }
    
    return graph(mesh.n_vertices(), mesh.edges(), node_features, edge_features);
}

sparse_matrix to_adjacency_matrix(const graph& graph) {
    validate(graph);
    std::vector<double> values;
    std::vector<std::size_t> col_indices;
    std::vector<std::size_t> row_ptr = {0};
    std::vector<std::vector<std::pair<std::size_t, double>>> rows(graph.num_vertices);
    for (const auto& [u, v] : graph.edges) {
        rows[u].emplace_back(v, 1.0);
        rows[v].emplace_back(u, 1.0);
    }
    
    for (auto& row : rows) {
        std::sort(std::execution::par_unseq, row.begin(), row.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [col, val] : row) {
            values.push_back(val);
            col_indices.push_back(col);
        }
    
        row_ptr.push_back(values.size());
    }
    
    return sparse_matrix(graph.num_vertices, graph.num_vertices,
                         std::move(values), std::move(col_indices), std::move(row_ptr));
}

feature_vector message_passing(
    const graph& graph,
    const feature_vector& messages,
    const vector& edge_weights) {
    validate(graph);
    if (messages.size() != graph.edges.size() || edge_weights.size() != graph.edges.size()) {
        throw std::runtime_error("message_passing: size mismatch");
    }
    if (!messages.empty() && !std::all_of(messages.begin(), messages.end(),
        [&](const auto& m) { return m.size() == messages[0].size(); })) {
        throw std::runtime_error("message_passing: inconsistent message dimensions");
    }
    
    feature_vector result(graph.num_vertices,
        messages.empty() ? vector() : vector(messages[0].size(), 0.0));
    
    std::vector<std::vector<vector>> local_results(graph.num_vertices, 
        messages.empty() ? std::vector<vector>() : std::vector<vector>(1, vector(messages[0].size(), 0.0)));
    
    std::for_each(std::execution::par_unseq, graph.edges.begin(), graph.edges.end(),
                  [&](const auto& edge) {
                      auto [u, v] = edge;
                      std::size_t edge_idx = &edge - graph.edges.data();
                      auto weighted_msg = gnnmath::vector::scalar_multiply(messages[edge_idx], edge_weights[edge_idx]);
                      local_results[u][0] = gnnmath::vector::operator+(local_results[u][0], weighted_msg);
                      local_results[v][0] = gnnmath::vector::operator+(local_results[v][0], weighted_msg);
                  });
    
    for (std::size_t i = 0; i < graph.num_vertices; ++i) {
        if (!local_results[i].empty()) {
            result[i] = local_results[i][0];
        }
    }
    
    return result;
}

feature_vector aggregate_features(
    const graph& graph,
    const feature_vector& node_features,
    const std::string& mode) {
    validate(graph);
    if (node_features.size() != graph.num_vertices) {
        throw std::runtime_error("aggregate_features: node_features size mismatch");
    }
    if (mode != "sum" && mode != "mean" && mode != "max") {
        throw std::runtime_error("aggregate_features: invalid mode");
    }
    if (!node_features.empty() && !std::all_of(node_features.begin(), node_features.end(),
        [&](const auto& f) { return f.size() == node_features[0].size(); })) {
        throw std::runtime_error("aggregate_features: inconsistent feature dimensions");
    }
    
    feature_vector result(graph.num_vertices,
        node_features.empty() ? vector() : vector(node_features[0].size(), 0.0));
    
    for (std::size_t v = 0; v < graph.num_vertices; ++v) {
        const auto& neighbors = graph.adjacency.at(v);
        if (neighbors.empty()) continue;
        
        if (mode == "sum" || mode == "mean") {
            for (const auto& [u, _] : neighbors) {
                result[v] = gnnmath::vector::operator+(result[v], node_features[u]);
            }
            
            if (mode == "mean") {
                result[v] = gnnmath::vector::scalar_multiply(result[v], 1.0 / neighbors.size());
            }
        } 
        else { // max
            result[v] = node_features[neighbors[0].first];
            for (const auto& [u, _] : neighbors) {
                for (std::size_t i = 0; i < result[v].size(); ++i) {
                    result[v][i] = std::max(result[v][i], node_features[u][i]);
                }
            }
        }
    }

    return result;
}

void update_features(graph& graph,
                     const feature_vector& node_features,
                     const feature_vector& edge_features) {
    if (node_features.size() != graph.num_vertices) {
        throw std::runtime_error("update_features: node_features size mismatch");
    }
    if (edge_features.size() != graph.edges.size()) {
        throw std::runtime_error("update_features: edge_features size mismatch");
    }
    if (!node_features.empty() && !std::all_of(node_features.begin(), node_features.end(),
        [&](const auto& f) { return f.size() == node_features[0].size(); })) {
        throw std::runtime_error("update_features: inconsistent node feature dimensions");
    }
    if (!edge_features.empty() && !std::all_of(edge_features.begin(), edge_features.end(),
        [&](const auto& f) { return f.size() == edge_features[0].size(); })) {
        throw std::runtime_error("update_features: inconsistent edge feature dimensions");
    }

    graph.node_features = node_features;
    graph.edge_features = edge_features;
}

std::vector<std::size_t> get_neighbors(const graph& graph, std::size_t vertex) {
    validate(graph);
    if (vertex >= graph.num_vertices) {
        throw std::runtime_error("get_neighbors: invalid vertex index");
    }
    
    std::vector<std::size_t> neighbors;
    if (auto it = graph.adjacency.find(vertex); it != graph.adjacency.end()) {
        for (const auto& [u, _] : it->second) {
            neighbors.push_back(u);
        }
    }
    
    return neighbors;
}

std::vector<std::size_t> compute_degree(const graph& graph) {
    validate(graph);
    std::vector<std::size_t> degrees(graph.num_vertices, 0);
    
    for (std::size_t v = 0; v < graph.num_vertices; ++v) {
        if (auto it = graph.adjacency.find(v); it != graph.adjacency.end()) {
            degrees[v] = it->second.size();
        }
    }
    
    return degrees;
}

sparse_matrix laplacian_matrix(const graph& graph) {
    validate(graph);
    auto degrees = compute_degree(graph);
    std::vector<double> values;
    std::vector<std::size_t> col_indices;
    std::vector<std::size_t> row_ptr = {0};
    std::vector<std::vector<std::pair<std::size_t, double>>> rows(graph.num_vertices);
    for (std::size_t v = 0; v < graph.num_vertices; ++v) {
        rows[v].emplace_back(v, static_cast<double>(degrees[v]));
    }
    
    for (const auto& [u, v] : graph.edges) {
        rows[u].emplace_back(v, -1.0);
        rows[v].emplace_back(u, -1.0);
    }
    
    for (auto& row : rows) {
        std::sort(std::execution::par_unseq, row.begin(), row.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [col, val] : row) {
            values.push_back(val);
            col_indices.push_back(col);
        }
    
        row_ptr.push_back(values.size());
    }
    
    return sparse_matrix(graph.num_vertices, graph.num_vertices,
                         std::move(values), std::move(col_indices), std::move(row_ptr));
}

} // namespace graph
} // namespace gnnmath