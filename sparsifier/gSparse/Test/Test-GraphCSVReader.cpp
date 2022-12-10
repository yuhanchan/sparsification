// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <gSparse/GraphCSVReader.hpp>
#include <gtest/gtest.h>
#include <memory>

/*******************************************************
 * Test Cases
 * ******************************************************/
TEST(GraphCSVReader, ReadEdge) {
  gSparse::PrecisionRowMatrix Weight;
  gSparse::EdgeMatrix Edges;
  std::shared_ptr<gSparse::IGraphReader> reader =
      std::make_shared<gSparse::GraphCSVReader>("test-edges.csv", "None", " ");
  reader->Read(Edges, Weight);
  // Building edges validation data
  gSparse::EdgeMatrix Edges_Validate(3, 2);
  Edges_Validate(0, 0) = 1;
  Edges_Validate(0, 1) = 2;
  Edges_Validate(1, 0) = 2;
  Edges_Validate(1, 1) = 3;
  Edges_Validate(2, 0) = 3;
  Edges_Validate(2, 1) = 1;

  // Building weight validation data
  gSparse::PrecisionRowMatrix Weight_Validate(3, 1);
  Weight_Validate << 1, 1, 1;
  EXPECT_EQ((Weight - Weight_Validate).norm(), 0.0);
  EXPECT_EQ((Edges - Edges_Validate).norm(), 0.0);
}

TEST(GraphCSVReader, ReadAll) {
  /* Testing Initialization */
  gSparse::GraphCSVReader csvReader("test-edges.csv", "test-weight.csv", " ");

  // Loading Test Data
  gSparse::EdgeMatrix Edges;
  gSparse::PrecisionRowMatrix Weight;
  csvReader.Read(Edges, Weight);

  // Building edges validation data
  gSparse::EdgeMatrix Edges_Validate(3, 2);
  Edges_Validate(0, 0) = 1;
  Edges_Validate(0, 1) = 2;
  Edges_Validate(1, 0) = 2;
  Edges_Validate(1, 1) = 3;
  Edges_Validate(2, 0) = 3;
  Edges_Validate(2, 1) = 1;

  // Building weight validation data
  gSparse::PrecisionRowMatrix Weight_Validate(3, 1);
  Weight_Validate << 1, 2, 3;

  EXPECT_EQ((Weight - Weight_Validate).norm(), 0.0);
  EXPECT_EQ((Edges - Edges_Validate).norm(), 0.0);
}
TEST(GraphCSVReader, EdgeNotFound) {
  gSparse::EdgeMatrix Edges;
  gSparse::PrecisionRowMatrix Weight;
  /* Test fail file opening */
  gSparse::GraphCSVReader csvReaderFail("ThisFileDoesNotExist.txt",
                                        "Test/test-weight.csv", " ");
  EXPECT_ANY_THROW(csvReaderFail.Read(Edges, Weight));
}
TEST(GraphCSVReader, WeightNotFound) {
  gSparse::EdgeMatrix Edges;
  gSparse::PrecisionRowMatrix Weight;
  /* Test fail file opening */
  gSparse::GraphCSVReader csvReaderFail("Test/test-edges.csv",
                                        "ThisFileDoesNotExist.txt", " ");
  EXPECT_ANY_THROW(csvReaderFail.Read(Edges, Weight));
}
TEST(GraphCSVReader, AllNotFound) {
  gSparse::EdgeMatrix Edges;
  gSparse::PrecisionRowMatrix Weight;
  /* Test fail file opening */
  gSparse::GraphCSVReader csvReaderFail("ThisFileDoesNotExist.txt",
                                        "ThisFileDoesNotExist.txt", " ");
  EXPECT_ANY_THROW(csvReaderFail.Read(Edges, Weight));
}
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
