#include <gtest/gtest.h>
#include <gnnmath/gnn/layers/layer.hpp>
#include <gnnmath/gnn/layers/gcn_layer.hpp>
#include <gnnmath/gnn/layers/edge_conv_layer.hpp>
#include <gnnmath/gnn/pipeline.hpp>
#include <gnnmath/gnn/training.hpp>
#include <gnnmath/math/dense_matrix.hpp>
#include <gnnmath/math/sparse_matrix.hpp>
#include <cmath>
#include <fstream>

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

TEST(AdamOptimizerTest, ShapeChangeReinitializesState) {
    adam_optimizer opt(0.01);

    // First update with a 2x3 layer at index 0
    matrix::dense_matrix w1(2, 3);
    vector::vector b1(3, 0.5);
    matrix::dense_matrix gw1(2, 3);
    vector::vector gb1(3, 0.1);
    gw1(0, 0) = 1.0;
    EXPECT_NO_THROW(opt.update(w1, b1, gw1, gb1, 0));

    // Reuse the same layer index with a larger 4x5 layer: moments must be
    // reallocated for the new shape, not indexed with the old one
    matrix::dense_matrix w2(4, 5);
    vector::vector b2(5, 0.5);
    matrix::dense_matrix gw2(4, 5);
    vector::vector gb2(5, 0.1);
    gw2(3, 4) = 1.0;
    EXPECT_NO_THROW(opt.update(w2, b2, gw2, gb2, 0));
    EXPECT_TRUE(matrix::is_valid(w2));
    for (double v : b2) {
        EXPECT_TRUE(std::isfinite(v));
    }

    // And shrinking again must also be safe
    matrix::dense_matrix w3(1, 2);
    vector::vector b3(2, 0.5);
    matrix::dense_matrix gw3(1, 2);
    vector::vector gb3(2, 0.1);
    EXPECT_NO_THROW(opt.update(w3, b3, gw3, gb3, 0));
    EXPECT_TRUE(matrix::is_valid(w3));
}

TEST(PipelineSerializationTest, FormatV2RoundTrip) {
    auto p = std::make_shared<pipeline>();
    p->add_layer(std::make_unique<gcn_layer>(3, 4, activation_type::GELU));
    p->add_layer(std::make_unique<edge_conv_layer>(4, 2, activation_type::SIGMOID));

    std::string path = ::testing::TempDir() + "meshgnn_v2_roundtrip.bin";
    p->save(path);

    // Reconstruct from the file alone
    pipeline p2 = pipeline::load_new(path);
    ASSERT_EQ(p2.num_layers(), 2u);
    EXPECT_EQ(p2.layers()[0]->kind(), layer_kind::gcn);
    EXPECT_EQ(p2.layers()[0]->act(), activation_type::GELU);
    EXPECT_EQ(p2.layers()[1]->kind(), layer_kind::edge_conv);
    EXPECT_EQ(p2.layers()[1]->act(), activation_type::SIGMOID);
    EXPECT_EQ(p2.layers()[0]->in_features(), 3u);
    EXPECT_EQ(p2.layers()[1]->out_features(), 2u);

    // Identical outputs
    auto adj = matrix::build_adj_matrix(3, {{0, 1}, {1, 2}, {2, 0}});
    std::vector<vector::vector> features = {
        {1.0, 0.0, 0.5}, {0.0, 1.0, -0.5}, {1.0, 1.0, 0.0}
    };
    auto out1 = p->process(features, adj);
    auto out2 = p2.process(features, adj);
    ASSERT_EQ(out1.size(), out2.size());
    for (std::size_t i = 0; i < out1.size(); ++i) {
        for (std::size_t j = 0; j < out1[i].size(); ++j) {
            EXPECT_NEAR(out1[i][j], out2[i][j], 1e-12);
        }
    }
}

TEST(PipelineSerializationTest, FormatV1BackCompat) {
    // Hand-craft a v1 file: magic, version 1, one GCN layer 2x3
    std::string path = ::testing::TempDir() + "meshgnn_v1_compat.bin";
    {
        std::ofstream f(path, std::ios::binary);
        uint32_t magic = 0x4D475050, version = 1, num_layers = 1;
        uint8_t type = 1;
        uint32_t rows = 2, cols = 3, bias_size = 3;
        f.write(reinterpret_cast<const char*>(&magic), 4);
        f.write(reinterpret_cast<const char*>(&version), 4);
        f.write(reinterpret_cast<const char*>(&num_layers), 4);
        f.write(reinterpret_cast<const char*>(&type), 1);
        f.write(reinterpret_cast<const char*>(&rows), 4);
        f.write(reinterpret_cast<const char*>(&cols), 4);
        for (int i = 0; i < 6; ++i) {
            double v = 0.5 * (i + 1);
            f.write(reinterpret_cast<const char*>(&v), 8);
        }
        f.write(reinterpret_cast<const char*>(&bias_size), 4);
        for (int i = 0; i < 3; ++i) {
            double v = -0.25 * (i + 1);
            f.write(reinterpret_cast<const char*>(&v), 8);
        }
    }

    // v1 into a matching prebuilt pipeline works
    pipeline p;
    p.add_layer(std::make_unique<gcn_layer>(2, 3));
    EXPECT_NO_THROW(p.load(path));
    auto* gcn = dynamic_cast<gcn_layer*>(p.layers()[0].get());
    ASSERT_NE(gcn, nullptr);
    EXPECT_DOUBLE_EQ(gcn->weights()(0, 0), 0.5);
    EXPECT_DOUBLE_EQ(gcn->weights()(1, 2), 3.0);
    EXPECT_DOUBLE_EQ(gcn->bias()[2], -0.75);

    // v1 into an empty pipeline is rejected
    pipeline empty;
    EXPECT_THROW(empty.load(path), std::runtime_error);
}
