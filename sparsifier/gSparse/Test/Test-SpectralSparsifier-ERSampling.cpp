// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#include <gtest/gtest.h>
#include <gSparse/UndirectedGraph.hpp>
#include <gSparse/SpectralSparsifier/ERSampling.hpp>

#include <iostream>
TEST(ERSampling,ApproximateER)
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
    gSparse::SpectralSparsifier::ERSampling sparsifier(test);
    EXPECT_NO_THROW(sparsifier.Compute());
    EXPECT_NO_THROW(sparsifier.GetSparsifiedGraph());
}
TEST(ERSampling,ExactER)
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
    gSparse::SpectralSparsifier::ERSampling sparsifier(test);
    sparsifier.SetERPolicy(gSparse::SpectralSparsifier::ER_METHODS::EXACT_ER);
    EXPECT_NO_THROW(sparsifier.Compute());
    EXPECT_NO_THROW(sparsifier.GetSparsifiedGraph());

}

TEST(ERSampling,Getters)
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
    gSparse::SpectralSparsifier::ERSampling sparsifier(test,5.0,0.5, gSparse::SpectralSparsifier::EXACT_ER);
    EXPECT_EQ(0.5,sparsifier.GetEpsilon());
    EXPECT_EQ(5.0,sparsifier.GetC());
    EXPECT_EQ(gSparse::SpectralSparsifier::EXACT_ER, sparsifier.GetERPolicy());
}

TEST(ERSampling,Setters)
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
    gSparse::SpectralSparsifier::ERSampling sparsifier(test);
    
    EXPECT_EQ(0.3f,sparsifier.GetEpsilon());
    EXPECT_EQ(4.0f,sparsifier.GetC());
    EXPECT_EQ(gSparse::SpectralSparsifier::APPROXIMATE_ER,sparsifier.GetERPolicy());
    sparsifier.SetC(5.0);
    sparsifier.SetERPolicy(gSparse::SpectralSparsifier::EXACT_ER);
    sparsifier.SetEpsilon(0.5);
    EXPECT_EQ(0.5,sparsifier.GetEpsilon());
    EXPECT_EQ(5.0,sparsifier.GetC());
    EXPECT_EQ(gSparse::SpectralSparsifier::EXACT_ER,sparsifier.GetERPolicy());

}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}