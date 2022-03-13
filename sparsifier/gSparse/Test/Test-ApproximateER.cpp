// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#include <gtest/gtest.h>
#include <gSparse/UndirectedGraph.hpp>
#include <gSparse/ER/ApproximateER.hpp>

#include <iostream>
TEST(ApproximateER,JACOBI_CG)
{
    // Call constructors
    gSparse::EdgeMatrix Edges(3, 2);
	gSparse::PrecisionMatrix Weights(3, 1);
	Edges(0, 0) = 0; Edges(0, 1) = 1;
	Edges(1, 0) = 1; Edges(1, 1) = 2;
	Edges(2, 0) = 2; Edges(2, 1) = 3;
	Weights << 1, 2, 3;

    // Call constructors
    gSparse::Graph test(new gSparse::UndirectedGraph(Edges, Weights));
    gSparse::ER::ApproximateER testPolicy;
    gSparse::PrecisionRowMatrix er;
    EXPECT_NO_THROW(testPolicy.CalculateER(er,test));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}