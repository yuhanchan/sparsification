#ifndef GSPARSE_BUILDER_COMPLETEGRAPH_HPP
#define GSPARSE_BUILDER_COMPLETEGRAPH_HPP

#include "../Config.hpp"
#include "../UndirectedGraph.hpp"
#include "../Util/Sampling.hpp"

#include <random>

namespace gSparse
{
	namespace Builder
	{
        /// \ingroup Builder
        ///
        /// This function builds a unit complete graph 
        /// \param nodeCount  Number of nodes in the Complete graph 
        ///
		inline gSparse::Graph buildUnitCompleteGraph(std::size_t nodeCount)
		{
            std::size_t & n = nodeCount;

			gSparse::EdgeMatrix resultEdge = gSparse::EdgeMatrix((n * (n - 1)) / 2, 2);
			std::size_t row = 0;
			for (std::size_t i = 0; i != n; ++i)
			{
				for (std::size_t j = 0; j != n; ++j)
				{
					if (i < j)
					{
						resultEdge(row, 0) = i;
						resultEdge(row, 1) = j;
						++row;
					}
				}
			}
			return std::make_shared<gSparse::UndirectedGraph>(resultEdge);
		}
        /// \ingroup Builder
        ///
        /// This function builds a unit complete graph 
        /// \param nodeCount  Number of nodes in the Complete graph 
        /// \param lower_weight lower bound of the weight to randomize. The value must be greater than zero.
        /// \param upper_weight upper bound of the weight to randomize. The value must be greater than zero.
        inline gSparse::Graph buildRandomCompleteGraph(std::size_t nodeCount, double lower_weight, double upper_weight)
		{
            std::size_t & n = nodeCount;

            #ifndef NDEBUG
			assert (upper_weight >= lower_weight);
            assert (lower_weight > 0.0f);
            assert (upper_weight > 0.0f);
			#endif 
            
			gSparse::EdgeMatrix resultEdge = gSparse::EdgeMatrix((n * (n - 1)) / 2, 2);
            gSparse::PrecisionRowMatrix resultWeight = gSparse::PrecisionRowMatrix((n * (n - 1)) / 2, 1);
			std::size_t row = 0;
            std::uniform_real_distribution<double> dist(lower_weight, upper_weight);
			for (std::size_t i = 0; i != n; ++i)
			{
				for (std::size_t j = 0; j != n; ++j)
				{
					if (i < j)
					{
						resultEdge(row, 0) = i;
						resultEdge(row, 1) = j;
                        resultWeight(row) = gSparse::Util::sample(dist);
						++row;
					}
				}
			}
			return std::make_shared<gSparse::UndirectedGraph>(resultEdge,resultWeight);
		}
	}
}

#endif
