#include <gtest/gtest.h>
#include <gnnmath/mesh.hpp>
#include <cmath>

using namespace gnnmath::mesh;

class MeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple triangle mesh programmatically
        m.vertices().push_back({0.0, 0.0, 0.0});
        m.vertices().push_back({1.0, 0.0, 0.0});
        m.vertices().push_back({0.5, 1.0, 0.0});
        m.faces().push_back({0, 1, 2});

        // Build edges
        auto add_edge = [this](std::size_t u, std::size_t v) {
            auto key = std::make_pair(std::min(u, v), std::max(u, v));
            if (m.edge_ind_map().find(key) == m.edge_ind_map().end()) {
                std::size_t idx = m.edges().size();
                m.edges().push_back({u, v});
                m.edge_ind_map()[key] = idx;
                m.adjacency()[u].push_back(v);
                m.adjacency()[v].push_back(u);
                m.incident_edges()[u].push_back(idx);
                m.incident_edges()[v].push_back(idx);
            }
        };
        add_edge(0, 1);
        add_edge(1, 2);
        add_edge(2, 0);
    }

    mesh m;
};

TEST_F(MeshTest, BasicProperties) {
    EXPECT_EQ(m.n_vertices(), 3);
    EXPECT_EQ(m.n_edges(), 3);
    EXPECT_EQ(m.n_faces(), 1);
}

TEST_F(MeshTest, IsValid) {
    EXPECT_TRUE(m.is_valid());
}

TEST_F(MeshTest, Validate) {
    EXPECT_NO_THROW(m.validate());
}

TEST_F(MeshTest, InvalidMeshEmpty) {
    mesh empty;
    EXPECT_FALSE(empty.is_valid());
}

TEST_F(MeshTest, GetNeighbors) {
    auto neighbors = m.get_neighbors(0);
    EXPECT_EQ(neighbors.size(), 2);
    // Should contain 1 and 2
    bool has_1 = std::find(neighbors.begin(), neighbors.end(), 1) != neighbors.end();
    bool has_2 = std::find(neighbors.begin(), neighbors.end(), 2) != neighbors.end();
    EXPECT_TRUE(has_1);
    EXPECT_TRUE(has_2);
}

TEST_F(MeshTest, GetNeighborsInvalidVertex) {
    EXPECT_THROW(m.get_neighbors(100), std::runtime_error);
}

TEST_F(MeshTest, GetIncidentEdges) {
    auto edges = m.get_incident_edges(0);
    EXPECT_EQ(edges.size(), 2);
}

TEST_F(MeshTest, ComputeNodeFeatures) {
    auto features = m.compute_node_features();
    ASSERT_EQ(features.size(), 3);
    ASSERT_EQ(features[0].size(), 3);
    EXPECT_DOUBLE_EQ(features[0][0], 0.0);
    EXPECT_DOUBLE_EQ(features[1][0], 1.0);
}

TEST_F(MeshTest, ComputeEdgeFeatures) {
    auto features = m.compute_edge_features();
    ASSERT_EQ(features.size(), 3);
    // Edge 0-1 has length 1.0
    EXPECT_NEAR(features[0][0], 1.0, 1e-10);
}

TEST_F(MeshTest, ToAdjacencyMatrix) {
    auto adj = m.to_adjacency_matrix();
    EXPECT_EQ(adj.rows, 3);
    EXPECT_EQ(adj.cols, 3);
}

TEST_F(MeshTest, ComputeNormals) {
    auto normals = m.compute_normals();
    ASSERT_EQ(normals.size(), 3);
    ASSERT_EQ(normals[0].size(), 3);
    // For a flat triangle in XY plane, normals should be along Z
    for (const auto& n : normals) {
        EXPECT_NEAR(std::abs(n[2]), 1.0, 0.01);
    }
}

TEST_F(MeshTest, SampleVertices) {
    auto samples = m.sample_vertices(2);
    EXPECT_EQ(samples.size(), 2);
    // All samples should be valid vertex indices
    for (auto idx : samples) {
        EXPECT_LT(idx, m.n_vertices());
    }
}

TEST_F(MeshTest, SampleVerticesTooMany) {
    EXPECT_THROW(m.sample_vertices(10), std::invalid_argument);
}

TEST_F(MeshTest, AddVertexNoise) {
    auto original = m.vertices()[0];
    m.add_vertex_noise(0.01);
    // Vertices should be slightly different
    bool changed = false;
    for (std::size_t i = 0; i < 3; ++i) {
        if (std::abs(m.vertices()[0][i] - original[i]) > 1e-10) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

TEST_F(MeshTest, AddVertexNoiseNegativeScale) {
    EXPECT_THROW(m.add_vertex_noise(-1.0), std::invalid_argument);
}

class MeshSimplificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a larger mesh with 4 vertices (tetrahedron projection)
        m.vertices().push_back({0.0, 0.0, 0.0});
        m.vertices().push_back({1.0, 0.0, 0.0});
        m.vertices().push_back({0.5, 1.0, 0.0});
        m.vertices().push_back({0.5, 0.5, 1.0});

        // 4 faces
        m.faces().push_back({0, 1, 2});
        m.faces().push_back({0, 1, 3});
        m.faces().push_back({1, 2, 3});
        m.faces().push_back({0, 2, 3});

        // Build edges
        auto add_edge = [this](std::size_t u, std::size_t v) {
            auto key = std::make_pair(std::min(u, v), std::max(u, v));
            if (m.edge_ind_map().find(key) == m.edge_ind_map().end()) {
                std::size_t idx = m.edges().size();
                m.edges().push_back({u, v});
                m.edge_ind_map()[key] = idx;
                m.adjacency()[u].push_back(v);
                m.adjacency()[v].push_back(u);
                m.incident_edges()[u].push_back(idx);
                m.incident_edges()[v].push_back(idx);
            }
        };

        for (const auto& f : m.faces()) {
            add_edge(f[0], f[1]);
            add_edge(f[1], f[2]);
            add_edge(f[2], f[0]);
        }
    }

    mesh m;
};

TEST_F(MeshSimplificationTest, SimplifyTargetExceedsVertices) {
    EXPECT_THROW(simplify_gnn_edge_collapse(m, 10, {}), std::invalid_argument);
}

TEST_F(MeshSimplificationTest, SimplifyNoOp) {
    // Target equals current vertices - should be no-op
    simplify_gnn_edge_collapse(m, 4, {});
    EXPECT_EQ(m.n_vertices(), 4);
}
