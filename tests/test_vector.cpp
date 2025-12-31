#include <gtest/gtest.h>
#include <gnnmath/math/vector.hpp>
#include <cmath>

using namespace gnnmath::vector;

class VectorTest : public ::testing::Test {
protected:
    vector a = {1.0, 2.0, 3.0};
    vector b = {4.0, 5.0, 6.0};
};

TEST_F(VectorTest, Addition) {
    auto result = a + b;
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 5.0);
    EXPECT_DOUBLE_EQ(result[1], 7.0);
    EXPECT_DOUBLE_EQ(result[2], 9.0);
}

TEST_F(VectorTest, AdditionSizeMismatch) {
    vector c = {1.0, 2.0};
    EXPECT_THROW(a + c, std::runtime_error);
}

TEST_F(VectorTest, Subtraction) {
    auto result = b - a;
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 3.0);
    EXPECT_DOUBLE_EQ(result[1], 3.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST_F(VectorTest, ScalarMultiply) {
    auto result = scalar_multiply(a, 2.0);
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], 4.0);
    EXPECT_DOUBLE_EQ(result[2], 6.0);
}

TEST_F(VectorTest, DotProduct) {
    double result = dot_product(a, b);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_DOUBLE_EQ(result, 32.0);
}

TEST_F(VectorTest, EuclideanNorm) {
    vector v = {3.0, 4.0};
    double norm = euclidean_norm(v);
    EXPECT_DOUBLE_EQ(norm, 5.0);
}

TEST_F(VectorTest, ReLU) {
    vector v = {-2.0, -1.0, 0.0, 1.0, 2.0};
    auto result = relu(v);
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
    EXPECT_DOUBLE_EQ(result[3], 1.0);
    EXPECT_DOUBLE_EQ(result[4], 2.0);
}

TEST_F(VectorTest, Sigmoid) {
    vector v = {0.0};
    auto result = sigmoid(v);
    EXPECT_DOUBLE_EQ(result[0], 0.5);
}

TEST_F(VectorTest, SigmoidBounds) {
    vector v = {-100.0, 100.0};
    auto result = sigmoid(v);
    EXPECT_GT(result[0], 0.0);
    EXPECT_LT(result[0], 0.01);
    EXPECT_GT(result[1], 0.99);
    EXPECT_LT(result[1], 1.0);
}

TEST_F(VectorTest, Softmax) {
    vector v = {1.0, 1.0, 1.0};
    auto result = softmax(v);
    // All equal inputs should give uniform distribution
    EXPECT_NEAR(result[0], 1.0/3.0, 1e-10);
    EXPECT_NEAR(result[1], 1.0/3.0, 1e-10);
    EXPECT_NEAR(result[2], 1.0/3.0, 1e-10);
    // Sum should be 1
    double sum = result[0] + result[1] + result[2];
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST_F(VectorTest, SoftmaxEmpty) {
    vector v = {};
    EXPECT_THROW(softmax(v), std::runtime_error);
}

TEST_F(VectorTest, GELU) {
    vector v = {0.0};
    auto result = gelu(v);
    EXPECT_NEAR(result[0], 0.0, 1e-10);
}

TEST_F(VectorTest, Mish) {
    vector v = {0.0};
    auto result = mish(v);
    EXPECT_NEAR(result[0], 0.0, 1e-10);
}

TEST_F(VectorTest, Softsign) {
    vector v = {0.0, 1.0, -1.0};
    auto result = softsign(v);
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.5);
    EXPECT_DOUBLE_EQ(result[2], -0.5);
}
