#include <gtest/gtest.h>
#include <gnnmath/matrix.hpp>
#include <cmath>

using namespace gnnmath::matrix;

class DenseMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 2x3 matrix
        A = dense_matrix({{1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0}});
    }
    dense_matrix A{2, 3};
};

TEST_F(DenseMatrixTest, Construction) {
    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
}

TEST_F(DenseMatrixTest, ElementAccess) {
    EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(A(1, 1), 5.0);
}

TEST_F(DenseMatrixTest, ElementModification) {
    A(0, 0) = 10.0;
    EXPECT_DOUBLE_EQ(A(0, 0), 10.0);
}

TEST_F(DenseMatrixTest, OutOfBoundsAccess) {
    EXPECT_THROW(A(2, 0), std::out_of_range);
    EXPECT_THROW(A(0, 3), std::out_of_range);
}

TEST_F(DenseMatrixTest, MatrixVectorMultiply) {
    vector v = {1.0, 1.0, 1.0};
    auto result = matrix_vector_multiply(A, v);
    ASSERT_EQ(result.size(), 2);
    EXPECT_DOUBLE_EQ(result[0], 6.0);  // 1+2+3
    EXPECT_DOUBLE_EQ(result[1], 15.0); // 4+5+6
}

TEST_F(DenseMatrixTest, MatrixMultiply) {
    dense_matrix B({{1.0, 2.0},
                    {3.0, 4.0},
                    {5.0, 6.0}});
    auto C = A * B;
    ASSERT_EQ(C.rows(), 2);
    ASSERT_EQ(C.cols(), 2);
    // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    EXPECT_DOUBLE_EQ(C(0, 0), 22.0);
}

TEST_F(DenseMatrixTest, Transpose) {
    auto AT = transpose(A);
    ASSERT_EQ(AT.rows(), 3);
    ASSERT_EQ(AT.cols(), 2);
    EXPECT_DOUBLE_EQ(AT(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(AT(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(AT(2, 0), 3.0);
}

TEST_F(DenseMatrixTest, Addition) {
    dense_matrix B({{1.0, 1.0, 1.0},
                    {1.0, 1.0, 1.0}});
    auto C = A + B;
    EXPECT_DOUBLE_EQ(C(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(C(1, 2), 7.0);
}

TEST_F(DenseMatrixTest, Identity) {
    auto I3 = I(3);
    ASSERT_EQ(I3.rows(), 3);
    ASSERT_EQ(I3.cols(), 3);
    EXPECT_DOUBLE_EQ(I3(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(I3(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(I3(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(I3(0, 1), 0.0);
}

TEST_F(DenseMatrixTest, FrobeniusNorm) {
    dense_matrix M({{3.0, 0.0},
                    {4.0, 0.0}});
    double norm = frobenius_norm(M);
    EXPECT_DOUBLE_EQ(norm, 5.0);  // sqrt(9 + 16) = 5
}

class SparseMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple adjacency matrix for triangle: 0-1-2-0
        edges = {{0, 1}, {1, 2}, {2, 0}};
        adj = build_adj_matrix(3, edges);
    }
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    sparse_matrix adj{3, 3};
};

TEST_F(SparseMatrixTest, BuildAdjMatrix) {
    EXPECT_EQ(adj.rows, 3);
    EXPECT_EQ(adj.cols, 3);
}

TEST_F(SparseMatrixTest, SparseVectorMultiply) {
    vector v = {1.0, 1.0, 1.0};
    auto result = adj.multiply(v);
    // Each vertex has 2 neighbors, so each result element should be 2
    ASSERT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 2.0);
}

TEST_F(SparseMatrixTest, ComputeDegrees) {
    auto degrees = compute_degrees(adj);
    ASSERT_EQ(degrees.size(), 3);
    EXPECT_DOUBLE_EQ(degrees[0], 2.0);
    EXPECT_DOUBLE_EQ(degrees[1], 2.0);
    EXPECT_DOUBLE_EQ(degrees[2], 2.0);
}

TEST_F(SparseMatrixTest, SparseAddition) {
    auto result = adj + adj;
    // Each non-zero element should be doubled
    vector v = {1.0, 1.0, 1.0};
    auto prod = result.multiply(v);
    EXPECT_DOUBLE_EQ(prod[0], 4.0);
}

TEST_F(SparseMatrixTest, SparseSubtraction) {
    auto result = adj - adj;
    // All elements should be zero
    vector v = {1.0, 1.0, 1.0};
    auto prod = result.multiply(v);
    EXPECT_NEAR(prod[0], 0.0, 1e-10);
    EXPECT_NEAR(prod[1], 0.0, 1e-10);
    EXPECT_NEAR(prod[2], 0.0, 1e-10);
}

TEST_F(SparseMatrixTest, LaplacianMatrix) {
    auto L = laplacian_matrix(adj);
    // Laplacian diagonal should be degree
    // Laplacian off-diagonal should be -1 for neighbors
    vector v = {1.0, 1.0, 1.0};
    auto result = L.multiply(v);
    // L * 1 = (D - A) * 1 = D*1 - A*1 = degrees - neighbor_sum
    // For regular graph: 2 - 2 = 0 for each vertex
    EXPECT_NEAR(result[0], 0.0, 1e-10);
    EXPECT_NEAR(result[1], 0.0, 1e-10);
    EXPECT_NEAR(result[2], 0.0, 1e-10);
}

TEST_F(SparseMatrixTest, ToDense) {
    auto dense = to_dense(adj);
    EXPECT_EQ(dense.rows(), 3);
    EXPECT_EQ(dense.cols(), 3);
    // Diagonal should be 0
    EXPECT_DOUBLE_EQ(dense(0, 0), 0.0);
    // Check some connections
    EXPECT_DOUBLE_EQ(dense(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(dense(1, 0), 1.0);
}

TEST_F(SparseMatrixTest, IsSymmetric) {
    EXPECT_TRUE(is_symmetric(adj));
}

TEST_F(SparseMatrixTest, SparseIdentity) {
    auto I3 = Identity(3);
    EXPECT_EQ(I3.rows, 3);
    EXPECT_EQ(I3.cols, 3);
    vector v = {1.0, 2.0, 3.0};
    auto result = I3.multiply(v);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}
