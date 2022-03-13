// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef GSPARSE_UTIL_SAMPLING_HPP
#define GSPARSE_UTIL_SAMPLING_HPP

#include "../Config.hpp"
#include <random>

namespace gSparse
{
    namespace Util
    {
        /* 
		Sample a number based from given discrete distribution
		The pseudo random generateuses Mersenne Twister algorithm.
		*/
		inline int sample(std::discrete_distribution<> & distribution)
		{
			thread_local std::mt19937 engine(std::random_device{}());
			return distribution(engine);
		}

		/*
		Uniform Sampling
		*/
		inline double sample(std::uniform_real_distribution<> & distribution)
		{
			thread_local std::mt19937 engine(std::random_device{}());
			return distribution(engine);
		}
		inline double sample(std::uniform_int_distribution<> & distribution)
		{
			thread_local std::mt19937 engine(std::random_device{}());
			return distribution(engine);
		}
		
    }
}

#endif