// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

/**
 * Example-4.cpp
 * 
 * This example shows you how to perform Spectral Sparsification 
 * by Effective Weight Resistance over completed graph set.
 * 
 * */

#include <gSparse/gSparse.hpp>
#include <iostream>

int main()
{
    // Generate 100x100 Complete Graph
    auto graph = gSparse::Builder::buildUnitCompleteGraph(100);
    
    // Display original edge count
    std::cout<<"Original Edge Count: " << graph->GetEdgeCount() << std::endl;

    // Creating Sparsifier Object
    gSparse::SpectralSparsifier::ERSampling sparsifier(graph);
    // Set Hyper-parameters
    // Approximate the Effective Weight Resistance (Faster)
    // C = 4 and Epsilon = 0.5
    sparsifier.SetERPolicy(gSparse::SpectralSparsifier::APPROXIMATE_ER);
    sparsifier.SetC(4.0);
    sparsifier.SetEpsilon(0.5);
    // Compute Effective Weight Resistance
    sparsifier.Compute();
    // Perform Sparsification 
    auto sparseGraph1 = sparsifier.GetSparsifiedGraph();
    std::cout<<"Sparised Edge Count (ApproxER): " << sparseGraph1->GetEdgeCount() << std::endl;

    // Set ER Policy to Exact Effective Resistance instead of Approximation
    // This is expected to perform much slower.
    sparsifier.SetERPolicy(gSparse::SpectralSparsifier::EXACT_ER);
    sparsifier.Compute();
    auto sparseGraph2 = sparsifier.GetSparsifiedGraph();
    std::cout<<"Sparised Edge Count (ApproxER): " << sparseGraph2->GetEdgeCount() << std::endl;
    return EXIT_SUCCESS;
}


