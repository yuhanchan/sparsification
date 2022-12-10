// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <gSparse/UndirectedGraph.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <string>

/*******************************************************
 * Set up and utility functions
 * ******************************************************/

class DummyReader : public gSparse::IGraphReader {
public:
  DummyReader() {}
  void Read(gSparse::EdgeMatrix &Edges, gSparse::PrecisionRowMatrix &Weights) {

    Edges = gSparse::EdgeMatrix(3, 2);
    Weights = gSparse::PrecisionMatrix(3, 1);

    Edges(0, 0) = 0;
    Edges(0, 1) = 1;
    Edges(1, 0) = 1;
    Edges(1, 1) = 2;
    Edges(2, 0) = 2;
    Edges(2, 1) = 3;

    Weights << 1, 2, 3;
  }
};

/*******************************************************
 * Test Suite
 * ******************************************************/

// Disable this test case, as support for copy by value is disabled.
/*TEST(UndirectedGraph, AssignmentOperator)
{
    // Generate data
    gSparse::EdgeMatrix Edges(3, 2);
        gSparse::PrecisionMatrix Weights(3, 1);
        Edges(0, 0) = 0; Edges(0, 1) = 1;
        Edges(1, 0) = 1; Edges(1, 1) = 2;
        Edges(2, 0) = 2; Edges(2, 1) = 3;
        Weights << 1, 2, 3;

    gSparse::EdgeMatrix Edges_2(3, 2);
        gSparse::PrecisionMatrix Weights_2(3, 1);
        Edges(0, 0) = 0; Edges(0, 1) = 1;
        Edges(1, 0) = 2; Edges(1, 1) = 3;
        Edges(2, 0) = 4; Edges(2, 1) = 5;
        Weights_2 << 2, 2, 2;

    // Generate test objects
    gSparse::UndirectedGraph test_1(Edges, Weights);
    gSparse::UndirectedGraph test_2(Edges_2, Weights_2);

    // Test Assignment operator operator
    test_2 = test_1;
    EXPECT_EQ(test_2.GetWeightList(),test_1.GetWeightList());
    EXPECT_EQ(test_2.GetEdgeList(),test_1.GetEdgeList());
}
*/

TEST(UndirectedGraph, Constructor) {
  // Call constructors
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 2, 3;

  // Call constructors
  gSparse::UndirectedGraph test_1(Edges, Weights);
  gSparse::Graph test_2(new gSparse::UndirectedGraph(Edges, Weights));

  // Test constructors
  EXPECT_EQ(test_1.GetEdgeList(), test_2->GetEdgeList());
}

TEST(UndirectedGraph, BuildEdgeOnly) {
  // When weight is not specified, it should defaults to one.

  // Build validation data
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 1, 1;

  // Construct the graph
  gSparse::UndirectedGraph test(Edges);
  gSparse::UndirectedGraph validation(Edges, Weights);

  // Validates
  EXPECT_EQ(test.GetEdgeList(), validation.GetEdgeList());
  EXPECT_EQ(test.GetWeightList(), validation.GetWeightList());
}

// Test GraphReader object
TEST(UndirectedGraph, GraphReader) {
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 2, 3;
  gSparse::GraphReader dummyReader(new DummyReader);
  gSparse::UndirectedGraph test_1(dummyReader);
  gSparse::UndirectedGraph test_2(Edges, Weights);
  EXPECT_EQ(test_1.GetEdgeList(), test_2.GetEdgeList());
  EXPECT_EQ(test_1.GetWeightList(), test_2.GetWeightList());
}

TEST(UndirectedGraph, UnitGraph) {
  gSparse::EdgeMatrix Edges(3, 2);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  gSparse::UndirectedGraph test(Edges);
  EXPECT_EQ(test.GetWeightList(), gSparse::PrecisionMatrix::Ones(3, 1));
}

// Test if the edge list is initialized properly
TEST(UndirectedGraph, Edges) {
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  gSparse::UndirectedGraph test(Edges);

  EXPECT_EQ(Edges, test.GetEdgeList());
}

// Test if the weight list is initialized properly
TEST(UndirectedGraph, Weights) {
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 2, 3;
  gSparse::UndirectedGraph test(Edges, Weights);

  EXPECT_EQ(Edges, test.GetEdgeList());
  EXPECT_EQ(Weights, test.GetWeightList());
}

// Check if the count of edges and vertices are good
TEST(UndirectedGraph, Counts) {
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 2, 3;

  gSparse::UndirectedGraph test(Edges, Weights);
  EXPECT_EQ(3, test.GetEdgeCount());
  EXPECT_EQ(4, test.GetNodeCount());
}

TEST(UndirectedGraph, AdjacencyMatrix) {
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 2, 3;

  Eigen::MatrixXd ValidateAdjacencyMatrix(4, 4);
  ValidateAdjacencyMatrix << 0, 1, 0, 0, 1, 0, 2, 0, 0, 2, 0, 3, 0, 0, 3, 0;
  gSparse::UndirectedGraph test(Edges, Weights);
  EXPECT_EQ(ValidateAdjacencyMatrix, test.GetAdjacentMatrix().toDense());
}
TEST(UndirectedGraph, WeightMatrix) {
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 2, 3;

  gSparse::PrecisionMatrix ValidateWeightMatrix(3, 3);
  ValidateWeightMatrix << 1, 0, 0, 0, 2, 0, 0, 0, 3;

  gSparse::UndirectedGraph test(Edges, Weights);
  EXPECT_EQ(test.GetWeightMatrix().toDense(), ValidateWeightMatrix);
}
TEST(UndirectedGraph, DegreeMatrix) {
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 2, 3;

  gSparse::PrecisionMatrix ValidateDegreeMatrix(4, 4);
  ValidateDegreeMatrix << 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3;
  gSparse::UndirectedGraph test(Edges, Weights);
  EXPECT_EQ(test.GetDegreeMatrix().toDense(), ValidateDegreeMatrix);
}
TEST(UndirectedGraph, LaplacianMatrix) {
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 2, 3;

  gSparse::PrecisionMatrix ValidateWeightedLaplacian(4, 4);
  ValidateWeightedLaplacian << 1, -1, 0, 0, -1, 3, -2, 0, 0, -2, 5, -3, 0, 0,
      -3, 3;
  gSparse::UndirectedGraph test(Edges, Weights);
  EXPECT_EQ(test.GetLaplacianMatrix().toDense(), ValidateWeightedLaplacian);
}
TEST(UndirectedGraph, IncidenceMatrix) {
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 2, 3;

  gSparse::PrecisionMatrix ValidateOrientedIncidentMatrix(3, 4);
  ValidateOrientedIncidentMatrix << 1, -1, 0, 0, 0, 1, -1, 0, 0, 0, 1, -1;
  gSparse::UndirectedGraph test(Edges, Weights);
  EXPECT_EQ(test.GetIncidentMatrix().toDense(), ValidateOrientedIncidentMatrix);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
