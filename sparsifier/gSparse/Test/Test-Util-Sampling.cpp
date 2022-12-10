// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#include <gSparse/Util/Sampling.hpp>
#include <gtest/gtest.h>

TEST(Util, Sampling) {
  // Just make sure that they run without crashing.
  // TODO: Find a way to test these random generators
  std::discrete_distribution<int> distribution{1, 2, 3};
  EXPECT_NO_THROW(gSparse::Util::sampleDiscreteDistribution(distribution));
}
