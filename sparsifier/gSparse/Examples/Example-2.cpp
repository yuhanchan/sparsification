// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

/**
 * Example-2.cpp
 *
 * The second examples shows how to create a Graph object from CSV file and
 * Eigen Matrix
 *
 * */

#include <gSparse/gSparse.hpp>
#include <iostream>
#include <memory>

int main() {

  // Create a CSV Reader and specify the file location
  gSparse::GraphReader csvReader = std::make_shared<gSparse::GraphCSVReader>(
      "Examples/Example-1-edges.csv", "Examples/Example-1-weights.csv");
  // Create an Undirected Graph. gSparse::Graph object is a shared_ptr.
  gSparse::Graph graph = std::make_shared<gSparse::UndirectedGraph>(csvReader);

  // Display the Edge list of the graph
  std::cout << "Edge List of Graph" << std::endl;
  std::cout << graph->GetEdgeList() << std::endl;
  // Display the weight list of the graph
  std::cout << "Weight List of Graph" << std::endl;
  std::cout << graph->GetWeightList() << std::endl;

  // Creating Edge and Weight Data Manually using Eigen Matrix
  gSparse::EdgeMatrix Edges(3, 2);
  gSparse::PrecisionMatrix Weights(3, 1);
  Edges(0, 0) = 0;
  Edges(0, 1) = 1;
  Edges(1, 0) = 1;
  Edges(1, 1) = 2;
  Edges(2, 0) = 2;
  Edges(2, 1) = 3;
  Weights << 1, 1, 1;

  // Construct the graph based on the data generated
  gSparse::UndirectedGraph test(Edges);
  gSparse::UndirectedGraph validation(Edges, Weights);

  return EXIT_SUCCESS;
}
