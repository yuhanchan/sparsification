// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

/**
 * Example-3.cpp
 * 
 * This examples shows how to calculate graph effective resistance.
 * 
 * */

#include <gSparse/gSparse.hpp>
#include <iostream>

int main()
{

    // Create a Unit Complete Graph
    auto graph = gSparse::Builder::buildUnitCompleteGraph(5);
    
    // Creating object to calculate Exact ER (slow)
    gSparse::ER::ExactER exactER;
    // Object for calculate Approximate ER (faster)
    gSparse::ER::ApproximateER approxER; 
    
    // Object to hold EffectiveResistance data
    gSparse::PrecisionRowMatrix er;
    
    // Calculate Exact ER
    exactER.CalculateER(er, graph );
    std::cout<<"Exact ER Results"<<std::endl;
    std::cout<<"----------------"<<std::endl;
    std::cout<<er<<std::endl;
    
    // Calculate Approximate ER
    approxER.CalculateER(er, graph);
    std::cout<<"Approx ER Results"<<std::endl;
    std::cout<<"----------------"<<std::endl;
    std::cout<<er<<std::endl;
    return EXIT_SUCCESS;
}


