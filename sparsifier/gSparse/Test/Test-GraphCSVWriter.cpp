// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <gSparse/GraphCSVWriter.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <string>

/*******************************************************
 * Set up and utility functions
 * ******************************************************/

// This function compares whether two files are the same
bool compareFiles(const std::string &p1, const std::string &p2) {
  std::ifstream f1(p1, std::ifstream::binary | std::ifstream::ate);
  std::ifstream f2(p2, std::ifstream::binary | std::ifstream::ate);
  if (f1.fail() || f2.fail()) {
    return false; // file problem
  }
  // seek back to beginning and use std::equal to compare contents
  f1.seekg(0, std::ifstream::beg);
  f2.seekg(0, std::ifstream::beg);
  return std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
                    std::istreambuf_iterator<char>(),
                    std::istreambuf_iterator<char>(f2.rdbuf()));
}

/* Dummy graph for testing
    TODO: Could leverage Google Mock for this
*/
class DummyGraph : public gSparse::IGraph {
private:
  gSparse::SparsePrecisionMatrix dummy;
  gSparse::EdgeMatrix dummyEdge;
  gSparse::PrecisionRowMatrix dummyWeight;

public:
  DummyGraph() {
    dummyEdge = gSparse::EdgeMatrix(3, 2);
    dummyWeight = gSparse::PrecisionRowMatrix(3, 1);
    dummyEdge(0, 0) = 1;
    dummyEdge(0, 1) = 2;
    dummyEdge(1, 0) = 2;
    dummyEdge(1, 1) = 3;
    dummyEdge(2, 0) = 3;
    dummyEdge(2, 1) = 1;
    dummyWeight << 1, 2, 3;
  }
  /* Get Interface*/
  const gSparse::SparsePrecisionMatrix &GetAdjacentMatrix() const {
    return dummy;
  }
  const gSparse::SparsePrecisionMatrix &GetIncidentMatrix() const {
    return dummy;
  }
  const gSparse::SparsePrecisionMatrix &GetDegreeMatrix() const {
    return dummy;
  }
  const gSparse::SparsePrecisionMatrix &GetLaplacianMatrix() const {
    return (dummy);
  }
  const gSparse::SparsePrecisionMatrix &GetWeightMatrix() const {
    return dummy;
  }
  const gSparse::EdgeMatrix &GetEdgeList() const {
    return dummyEdge;
    ;
  }
  const gSparse::PrecisionRowMatrix &GetWeightList() const {
    return dummyWeight;
  }
  std::size_t GetEdgeCount() const { return 0; }
  std::size_t GetNodeCount() const { return 0; }
};

/*******************************************************
 * Test Suites
 * ******************************************************/
TEST(GraphCSVWriter, Write1) {
  /* Testing Initialization */
  gSparse::GraphCSVWriter csvWriter("csvwriter-edges.csv",
                                    "csvwriter-weight.csv", " ");

  // Building Validation data
  gSparse::EdgeMatrix Edges(3, 2);
  Edges(0, 0) = 1;
  Edges(0, 1) = 2;
  Edges(1, 0) = 2;
  Edges(1, 1) = 3;
  Edges(2, 0) = 3;
  Edges(2, 1) = 1;

  // Building weight validation data
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Weight(3, 1);
  Weight << 1, 2, 3;

  // Write the file
  csvWriter.Write(Edges, Weight);

  // Compare if they are the same
  EXPECT_EQ(true, compareFiles("csvwriter-edges.csv", "test-edges.csv"));
  EXPECT_EQ(true, compareFiles("csvwriter-weight.csv", "test-weight.csv"));
}

TEST(GraphCSVWriter, Write2) {
  /* Testing Initialization */
  gSparse::GraphWriter csvWriter = std::make_shared<gSparse::GraphCSVWriter>(
      "csvwriter-edges.csv", "csvwriter-weight.csv", " ");

  // Building Validation data
  gSparse::EdgeMatrix Edges(3, 2);
  Edges(0, 0) = 1;
  Edges(0, 1) = 2;
  Edges(1, 0) = 2;
  Edges(1, 1) = 3;
  Edges(2, 0) = 3;
  Edges(2, 1) = 1;

  // Building weight validation data

  gSparse::PrecisionMatrix Weight(3, 1);
  Weight << 1, 2, 3;

  // Write the file
  csvWriter->Write(Edges, Weight);
  // Compare if they are the same
  EXPECT_EQ(true, compareFiles("csvwriter-edges.csv", "test-edges.csv"));
  EXPECT_EQ(true, compareFiles("csvwriter-weight.csv", "test-weight.csv"));
}

TEST(GraphCSVWriter, WriteGraphObject) {
  /* Testing Initialization */
  gSparse::GraphWriter csvWriter = std::make_shared<gSparse::GraphCSVWriter>(
      "csvwriter-edges.csv", "csvwriter-weight.csv", " ");

  /* Write file based on data from graph */
  gSparse::Graph dummy(new DummyGraph());
  csvWriter->Write(dummy);
  EXPECT_EQ(true, compareFiles("csvwriter-edges.csv", "test-edges.csv"));
  EXPECT_EQ(true, compareFiles("csvwriter-weight.csv", "test-weight.csv"));
}
TEST(GraphCSVWriter, WriteEdgeOnly) {
  /* Testing Initialization */
  gSparse::GraphWriter csvWriter = std::make_shared<gSparse::GraphCSVWriter>(
      "csvwriter-edges.csv", "None", " ");

  // Building Validation data
  gSparse::EdgeMatrix Edges(3, 2);
  Edges(0, 0) = 1;
  Edges(0, 1) = 2;
  Edges(1, 0) = 2;
  Edges(1, 1) = 3;
  Edges(2, 0) = 3;
  Edges(2, 1) = 1;

  // Write the file
  csvWriter->Write(Edges);
  EXPECT_EQ(true, compareFiles("csvwriter-edges.csv", "test-edges.csv"));
}

TEST(GraphCSVWriter, WriteGraphError1) {
  /* Testing Initialization */
  gSparse::GraphWriter csvWriter = std::make_shared<gSparse::GraphCSVWriter>(
      "csvwriter-edges.csv", "None", " ");

  /* Write file based on data from graph */
  gSparse::Graph dummy(new DummyGraph());
  EXPECT_ANY_THROW(csvWriter->Write(dummy));
}
TEST(GraphCSVWriter, WriteGraphError2) {
  /* Testing Initialization */
  gSparse::GraphWriter csvWriter = std::make_shared<gSparse::GraphCSVWriter>(
      "csvwriter-edges.csv", "None", " ");

  /* Write file based on data from graph */
  gSparse::Graph dummy;
  EXPECT_ANY_THROW(csvWriter->Write(dummy));
}

TEST(GraphCSVWriter, WriteGraphError3) {
  /* Testing Initialization */
  gSparse::GraphWriter csvWriter = std::make_shared<gSparse::GraphCSVWriter>(
      "csvwriter-edges.csv", "None", " ");

  // Building Validation data
  gSparse::EdgeMatrix Edges(3, 2);
  Edges(0, 0) = 1;
  Edges(0, 1) = 2;
  Edges(1, 0) = 2;
  Edges(1, 1) = 3;
  Edges(2, 0) = 3;
  Edges(2, 1) = 1;

  // Building weight validation data

  gSparse::PrecisionMatrix Weight(3, 1);
  Weight << 1, 2, 3;

  // Write the file, expects error
  EXPECT_ANY_THROW(csvWriter->Write(Edges, Weight));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
