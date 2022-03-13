// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#include <gtest/gtest.h>
#include <gSparse/gSparse.hpp>

TEST(Builder, CompleteUnitGraph)
{
	gSparse::PrecisionMatrix completeGraph(4, 4);
	completeGraph << 3, -1, -1, -1,
		-1, 3, -1, -1,
		-1, -1, 3, -1,
		-1, -1, -1, 3;
	EXPECT_EQ(completeGraph, gSparse::Builder::buildUnitCompleteGraph(4)->GetLaplacianMatrix().toDense());
}

TEST(Builder, CompleteRandomGraph)
{
	gSparse::PrecisionMatrix completeGraph(4, 4);
	completeGraph << 3, -1, -1, -1,
		-1, 3, -1, -1,
		-1, -1, 3, -1,
		-1, -1, -1, 3;
	EXPECT_EQ(completeGraph, gSparse::Builder::buildRandomCompleteGraph(4,1.0,1.0)->GetLaplacianMatrix().toDense());
	//std::cout<<gSparse::Builder::buildRandomCompleteGraph(4,0.0,5.0)->GetWeightList()<<std::endl;
}

