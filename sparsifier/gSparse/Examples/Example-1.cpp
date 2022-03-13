// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

/**
 * Example-1.cpp
 * 
 * The first examples shows how to write a unit graph into a CSV file.
 * 
 * */

#include <gSparse/gSparse.hpp>
#include <iostream>

int main()
{
    // Creates a random complete graph
    auto graph = gSparse::Builder::buildRandomCompleteGraph(4, 1.0, 5.0);

    // Create a Graph Writer for CSV file, specify edge and weight files to write to
    gSparse::GraphCSVWriter csvWriter("Examples/Example-1-edges.csv", "Examples/Example-1-weights.csv");

    // Write graph data into files
    csvWriter.Write(graph);

    return EXIT_SUCCESS;
}