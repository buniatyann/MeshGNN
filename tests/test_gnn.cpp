#include <gtest/gtest.h>
#include <gnnmath/gnn/layer.hpp>
#include <gnnmath/gnn/pipeline.hpp>
#include <gnnmath/gnn/training.hpp>
#include <gnnmath/matrix.hpp>
#include <cmath>

using namespace gnnmath;
using namespace gnnmath::gnn;

class GNNLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple 3-node graph (triangle)
        edges = {{0, 1}, {1, 2}, {2, 0}};
        adj = matrix::build_adj_matrix(3, edges);

        // 3 nodes, 2 features each
        features = {
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
        };
    }

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    matrix::sparse_matrix adj{3, 3};
    std::vector<vector::vector> features;
};

TEST_F(GNNLayerTest, GCNLayerConstruction) {
    EXPECT_NO_THROW(gcn_layer(2, 4));
}

TEST_F(GNNLayerTest, GCNLayerZeroDimensions) {
    EXPECT_THROW(gcn_layer(0, 4), std::runtime_error);
    EXPECT_THROW(gcn_layer(2, 0), std::runtime_error);
}

TEST_F(GNNLayerTest, GCNLayerForward) {
    gcn_layer layer(2, 4);
    auto output = layer.forward(features, adj);

    ASSERT_EQ(output.size(), 3);
    ASSERT_EQ(output[0].size(), 4);

    // Check outputs are finite
    for (const auto& row : output) {
        for (double val : row) {
            EXPECT_TRUE(std::isfinite(val));
        }
    }
}

TEST_F(GNNLayerTest, GCNLayerDimensionMismatch) {
    gcn_layer layer(5, 4);  // Expects 5 input features
    EXPECT_THROW(layer.forward(features, adj), std::runtime_error);
}

TEST_F(GNNLayerTest, EdgeConvLayerConstruction) {
    EXPECT_NO_THROW(edge_conv_layer(2, 4));
}

TEST_F(GNNLayerTest, EdgeConvLayerForward) {
    edge_conv_layer layer(2, 4);
    auto output = layer.forward(features, adj);

    ASSERT_EQ(output.size(), 3);
    ASSERT_EQ(output[0].size(), 4);

    // Check outputs are finite
    for (const auto& row : output) {
        for (double val : row) {
            EXPECT_TRUE(std::isfinite(val));
        }
    }
}

TEST_F(GNNLayerTest, LayerDimensions) {
    gcn_layer layer(2, 4);
    EXPECT_EQ(layer.in_features(), 2);
    EXPECT_EQ(layer.out_features(), 4);
}

TEST_F(GNNLayerTest, WeightsAccessible) {
    gcn_layer layer(2, 4);
    auto& weights = layer.weights();
    EXPECT_EQ(weights.rows(), 2);
    EXPECT_EQ(weights.cols(), 4);
}

TEST_F(GNNLayerTest, BiasAccessible) {
    gcn_layer layer(2, 4);
    auto& bias = layer.bias();
    EXPECT_EQ(bias.size(), 4);
}

class PipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        edges = {{0, 1}, {1, 2}, {2, 0}};
        adj = matrix::build_adj_matrix(3, edges);
        features = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
    }

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    matrix::sparse_matrix adj{3, 3};
    std::vector<vector::vector> features;
};

TEST_F(PipelineTest, EmptyPipeline) {
    pipeline p;
    EXPECT_EQ(p.num_layers(), 0);
}

TEST_F(PipelineTest, AddLayer) {
    pipeline p;
    p.add_layer(std::make_unique<gcn_layer>(3, 4));
    EXPECT_EQ(p.num_layers(), 1);
}

TEST_F(PipelineTest, AddNullLayer) {
    pipeline p;
    EXPECT_THROW(p.add_layer(nullptr), std::runtime_error);
}

TEST_F(PipelineTest, AddLayerDimensionMismatch) {
    pipeline p;
    p.add_layer(std::make_unique<gcn_layer>(3, 4));
    // Next layer expects 4 input features
    EXPECT_THROW(p.add_layer(std::make_unique<gcn_layer>(5, 2)), std::runtime_error);
}

TEST_F(PipelineTest, ProcessFeatures) {
    pipeline p;
    p.add_layer(std::make_unique<gcn_layer>(3, 4));
    p.add_layer(std::make_unique<gcn_layer>(4, 2));

    auto output = p.process(features, adj);
    ASSERT_EQ(output.size(), 3);
    ASSERT_EQ(output[0].size(), 2);
}

TEST_F(PipelineTest, ProcessEmptyPipeline) {
    pipeline p;
    EXPECT_THROW(p.process(features, adj), std::runtime_error);
}

class TrainerTest : public ::testing::Test {
protected:
    void SetUp() override {
        edges = {{0, 1}, {1, 2}, {2, 0}};
        adj = matrix::build_adj_matrix(3, edges);
        features = {
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
        };
        target = {
            {0.5, 0.5},
            {0.5, 0.5},
            {0.5, 0.5}
        };

        p.add_layer(std::make_unique<gcn_layer>(2, 2));
    }

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    matrix::sparse_matrix adj{3, 3};
    std::vector<vector::vector> features;
    std::vector<vector::vector> target;
    pipeline p;
};

TEST_F(TrainerTest, Construction) {
    EXPECT_NO_THROW(trainer(&p, 0.01));
}

TEST_F(TrainerTest, NullPipeline) {
    EXPECT_THROW(trainer(nullptr, 0.01), std::runtime_error);
}

TEST_F(TrainerTest, InvalidLearningRate) {
    EXPECT_THROW(trainer(&p, 0.0), std::runtime_error);
    EXPECT_THROW(trainer(&p, -0.01), std::runtime_error);
}

TEST_F(TrainerTest, MSELoss) {
    trainer t(&p, 0.01);
    auto predicted = p.process(features, adj);
    double loss = t.mse_loss(predicted, target);
    EXPECT_GE(loss, 0.0);
    EXPECT_TRUE(std::isfinite(loss));
}

TEST_F(TrainerTest, MSELossDimensionMismatch) {
    trainer t(&p, 0.01);
    std::vector<vector::vector> bad_target = {{1.0}};
    auto predicted = p.process(features, adj);
    EXPECT_THROW(t.mse_loss(predicted, bad_target), std::runtime_error);
}

TEST_F(TrainerTest, CrossEntropyLoss) {
    trainer t(&p, 0.01);
    // Create probability-like predictions
    std::vector<vector::vector> probs = {
        {0.7, 0.3},
        {0.4, 0.6},
        {0.5, 0.5}
    };
    std::vector<vector::vector> labels = {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0}
    };
    double loss = t.cross_entropy_loss(probs, labels);
    EXPECT_GE(loss, 0.0);
    EXPECT_TRUE(std::isfinite(loss));
}

TEST_F(TrainerTest, TrainStepSGD) {
    trainer t(&p, 0.01, optimizer_type::SGD);

    // Get initial prediction
    auto initial_pred = p.process(features, adj);
    double initial_loss = t.mse_loss(initial_pred, target);

    // Perform training step
    EXPECT_NO_THROW(t.train_step(features, adj, target));

    // Loss should change (not necessarily decrease in one step)
    auto new_pred = p.process(features, adj);
    double new_loss = t.mse_loss(new_pred, target);
    EXPECT_NE(initial_loss, new_loss);
}

TEST_F(TrainerTest, TrainStepAdam) {
    trainer t(&p, 0.01, optimizer_type::ADAM);
    EXPECT_NO_THROW(t.train_step(features, adj, target));
}

TEST_F(TrainerTest, SetLearningRate) {
    trainer t(&p, 0.01);
    t.set_learning_rate(0.001);
    EXPECT_DOUBLE_EQ(t.learning_rate(), 0.001);
}

TEST_F(TrainerTest, WeightDecay) {
    trainer t(&p, 0.01, optimizer_type::SGD, 0.001);
    EXPECT_NO_THROW(t.train_step(features, adj, target));
}
