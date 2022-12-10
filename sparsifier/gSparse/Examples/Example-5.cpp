// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

/**
 * Example-5.cpp
 *
 * This example perform Spectral Sparsification, and use Spectra to calcualte
 * Eigenvalue of Graph Laplacians
 *
 * */

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <gSparse/gSparse.hpp>
#include <iostream>

Eigen::VectorXcd GetTopEigenValue(const gSparse::SparsePrecisionMatrix &M) {
  std::cout << "Calculating Eigen Value" << std::endl;
  using namespace Spectra;
  SparseGenMatProd<double> op(M);

  // Construct eigen solver object, requesting the largest three eigenvalues
  GenEigsSolver<double, LARGEST_MAGN, SparseGenMatProd<double>> eigs(&op, 3, 6);

  // Initialize and compute
  eigs.init();
  int nconv = eigs.compute();

  // Retrieve results
  Eigen::VectorXcd evalues;
  if (eigs.info() == SUCCESSFUL)
    evalues = eigs.eigenvalues();

  std::cout << "Eigenvalues found:\n" << evalues << std::endl;
  return evalues;
}
int main() {
  // Generate 100x100 Complete Graph
  auto graph = gSparse::Builder::buildUnitCompleteGraph(100);
  // Creating Sparsifier Object
  gSparse::SpectralSparsifier::ERSampling sparsifier(graph);
  // Set Hyper-parameters
  // Approximate the Effective Weight Resistance (Faster)
  // C = 100 and Epsilon = 0.5
  sparsifier.SetERPolicy(gSparse::SpectralSparsifier::APPROXIMATE_ER);
  sparsifier.SetC(100.0);
  sparsifier.SetEpsilon(0.5);
  // Compute Effective Weight Resistance using ApproxER
  sparsifier.Compute();
  // Get a sparsified graph
  auto sparseGraph1 = sparsifier.GetSparsifiedGraph();
  // Set to EXACT ER
  sparsifier.SetERPolicy(gSparse::SpectralSparsifier::EXACT_ER);
  // Re-calcuate ER using ExactER
  sparsifier.Compute();
  // Get a sparsified graph
  auto sparseGraph2 = sparsifier.GetSparsifiedGraph();

  // Use Spectra to calculate top Eigen values.
  std::cout << "Original Eigen Value " << std::endl;
  GetTopEigenValue(graph->GetLaplacianMatrix());
  std::cout << "---------------------------" << std::endl;
  std::cout << "Top Eigen Value for ApproxER " << std::endl;
  GetTopEigenValue(sparseGraph1->GetLaplacianMatrix());
  std::cout << "Top Eigen Value for ApproxER - Original" << std::endl;
  GetTopEigenValue(sparseGraph1->GetLaplacianMatrix() -
                   graph->GetLaplacianMatrix());
  std::cout << "---------------------------" << std::endl;
  std::cout << "Top Eigen Value for ExactER " << std::endl;
  GetTopEigenValue(sparseGraph2->GetLaplacianMatrix());
  std::cout << "Top Eigen Value for ApproxER - ExactER" << std::endl;
  GetTopEigenValue(sparseGraph1->GetLaplacianMatrix() -
                   graph->GetLaplacianMatrix());
  return 0;
}
