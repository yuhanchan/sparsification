// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#include <gtest/gtest.h>
#include <gSparse/Util/JL.hpp>


TEST(JL,ZeroTol)
{
    auto mat = gSparse::Util::randomProjectionMatrix(3,2,1.0,1.0);
    auto Validate = gSparse::PrecisionMatrix(3, 2);
	Validate(0, 0) = -1.0f; Validate(0, 1) = -1.0f;
	Validate(1, 0) = -1.0f; Validate(1, 1) = -1.0f;
	Validate(2, 0) = -1.0f; Validate(2, 1) = -1.0f;

    EXPECT_EQ(Validate, mat);
}
TEST(JL,MaxTol)
{
    auto mat = gSparse::Util::randomProjectionMatrix(3,2,1.0,0.0);
    auto Validate = gSparse::PrecisionMatrix(3, 2);
	Validate(0, 0) = 1.0f; Validate(0, 1) = 1.0f;
	Validate(1, 0) = 1.0f; Validate(1, 1) = 1.0f;
	Validate(2, 0) = 1.0f; Validate(2, 1) = 1.0f;

    EXPECT_EQ(Validate, mat);
}
TEST(JL,Scale)
{
    auto mat = gSparse::Util::randomProjectionMatrix(3,2,4.0,0.0);
    auto Validate = gSparse::PrecisionMatrix(3, 2);
	Validate(0, 0) = 0.5f; Validate(0, 1) = 0.5f;
	Validate(1, 0) = 0.5f; Validate(1, 1) = 0.5f;
	Validate(2, 0) = 0.5f; Validate(2, 1) = 0.5f;

    EXPECT_EQ(Validate, mat);
}
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}