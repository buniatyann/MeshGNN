#include <gtest/gtest.h>
#include <gnnmath/graph.hpp>
#include <gnnmath/math/sparse_matrix.hpp>
#include <cmath>

using namespace gnnmath;

namespace {

std::vector<vector::vector> make_features(std::size_t n, std::size_t dim) {
    return std::vector<vector::vector>(n, vector::vector(dim, 1.0));
}

} // namespace

TEST(GraphTest, ValidConstruction) {
    std::vector<std::pair<std::size_t, std::size_t>> edges = {{0, 1}, {1, 2}, {2, 0}};
    EXPECT_NO_THROW(graph::graph(3, edges, make_features(3, 2), make_features(3, 1)));
}

TEST(GraphTest, SelfLoopRejected) {
    std::vector<std::pair<std::size_t, std::size_t>> edges = {{0, 1}, {1, 1}};
    EXPECT_THROW(graph::graph(3, edges, make_features(3, 2), make_features(2, 1)),
                 std::runtime_error);
}

TEST(GraphTest, DuplicateEdgeRejected) {
    // Same edge twice, including the reversed orientation
    std::vector<std::pair<std::size_t, std::size_t>> edges = {{0, 1}, {1, 0}};
    EXPECT_THROW(graph::graph(3, edges, make_features(3, 2), make_features(2, 1)),
                 std::runtime_error);
}

TEST(GraphTest, AdjacencyMatrixValidAndSymmetric) {
    std::vector<std::pair<std::size_t, std::size_t>> edges = {{0, 1}, {1, 2}, {2, 0}, {2, 3}};
    graph::graph g(4, edges, make_features(4, 2), make_features(4, 1));
    auto adj = graph::to_adjacency_matrix(g);
    EXPECT_TRUE(matrix::is_symmetric(adj));
    EXPECT_EQ(adj.vals.size(), 2 * edges.size());
}

TEST(GraphTest, AggregateFeaturesIsolatedVertex) {
    // Vertex 3 is isolated; aggregation must not throw
    std::vector<std::pair<std::size_t, std::size_t>> edges = {{0, 1}, {1, 2}};
    graph::graph g(4, edges, make_features(4, 2), make_features(2, 1));
    std::vector<vector::vector> result;
    EXPECT_NO_THROW(result = graph::aggregate_features(g, make_features(4, 2), "mean"));
    ASSERT_EQ(result.size(), 4);
    EXPECT_DOUBLE_EQ(result[3][0], 0.0);
}
