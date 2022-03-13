// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef GSPARSE_INTERFACE_GRAPH_HPP
#define GSPARSE_INTERFACE_GRAPH_HPP

#include <cstddef>    // size_t
#include <memory>   //shared_ptr
#include <Eigen/Dense> // Eigen Matrix
#include "../Config.hpp" // Library configuration

namespace gSparse
{
    //!  An interface class for Graph object.
    /*!
        This class defines an interface for gSparse's Graph primitives.
    */
	class IGraph
	{
	public:
		//!  Pure virtual function to get Graph Adjacency Matrix
		virtual const gSparse::SparsePrecisionMatrix & GetAdjacentMatrix() const = 0;
        //!  Pure virtual function to get Graph Incident Matrix
		virtual const gSparse::SparsePrecisionMatrix & GetIncidentMatrix() const = 0;
        //!  Pure virtual function to get Graph Degree Matrix
		virtual const gSparse::SparsePrecisionMatrix & GetDegreeMatrix() const = 0;
        //!  Pure virtual function to get Graph Laplacian Matrix
		virtual const gSparse::SparsePrecisionMatrix & GetLaplacianMatrix() const = 0;
        //!  Pure virtual function to get Graph Weight Matrix
		virtual const gSparse::SparsePrecisionMatrix & GetWeightMatrix() const = 0;
        //!  Pure virtual function to get Graph Edge List Matrix
		virtual const gSparse::EdgeMatrix & GetEdgeList() const = 0;
        //!  Pure virtual function to get Graph Weight List Matrix
		virtual const gSparse::PrecisionRowMatrix & GetWeightList() const = 0;
        //!  Pure virtual function to get number of edges in the graph
		virtual std::size_t GetEdgeCount() const = 0;
        //!  Pure virtual function to get number of nodes in the graph
		virtual std::size_t GetNodeCount() const = 0;
		/* Default Destructor*/
		virtual ~IGraph() = default;
	protected:
        //!  Protected default constructor. 
		IGraph() = default;
	};
	typedef std::shared_ptr<IGraph> Graph;
}
#endif