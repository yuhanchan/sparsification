// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef GSPARSE_BUILDER_COMMUNITYGRAPH_HPP
#define GSPARSE_BUILDER_COMMUNITYGRAPH_HPP

#include "../Config.hpp"
#include "../UndirectedGraph.hpp"
#include "CompleteGraph.hpp"

#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

namespace gSparse
{
	namespace Builder
	{
		std::string stringCleaner(const std::string & str)
		{
			std::stringstream ss;
			bool space = false;
			for (int i = 0; i != str.size(); ++i)
			{
				if (i == 0)
				{
					if(str[i] == ' ' || str[i] == '\t')
					{
						space = true;
						continue;
					}
				}
				if (str[i] == ' ' || str[i] == '\t')
				{
					if (space == true)
						continue;
					else
					{
						space = true;
						ss << ' ';
					}
					space = true;
					
				}
				else
				{
					space = false;
					ss << str[i];
				}
			}
			return ss.str();
		}
		/// \ingroup Builder
        ///
        /// This function builds a random community graph and write to a file
		/// TODO: This function requires a clean-up / refactor.
        ///
		void createRandomGraph(int total_community = 3, int bridges = 4, int community_size = 30, const std::string & name="random.csv")
		{
			std::default_random_engine generator;

			for (int i = 1; i != 2; ++i)
			{
				std::stringstream ss;
				ss << name;
				std::ofstream myfile;
				myfile.open(ss.str());

				//Each community is currently a complete graph to represent a dense graph. This can be sparsify safety through uniform sampling.
				auto g = gSparse::Builder::buildUnitCompleteGraph(i * community_size);
				
				//Connecting the bridges via uniform sampling
				int offset = g->GetNodeCount();
				std::uniform_int_distribution<int> distribution(0, offset);

				for (int community_id = 0; community_id != total_community; ++community_id)
				{
					std::stringstream ss;
					// Build communities of complete graph
					ss << g->GetEdgeList().array() + offset * community_id ;
					myfile << stringCleaner(ss.str()) << std::endl;
				}
				// connect the communities
				for (int j = 0; j != total_community; ++j)
				{
					// Connect the graphs
					for (int k = 0; k != total_community; ++k)
					{
						if (k > j)
						{
							for (int l = 0; l != bridges; ++l)
							{
								std::stringstream ss;
								ss << j * offset + distribution(generator) << " " << k * offset + distribution(generator);
								myfile << stringCleaner(ss.str()) << std::endl;
							}
						}
					}
				}
				myfile.close();
			}
		}
	}
}

#endif
