// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef GSPARSE_INTERFACE_GRAPHREADER_HPP
#define GSPARSE_INTERFACE_GRAPHREADER_HPP

#include <cstddef>    // size_t
#include <Eigen/Dense> // Eigen Matrix
#include <memory>   //shared_ptr
#include "../Config.hpp" // Library configuration

namespace gSparse
{
    //!  An interface class for Graph Input.
    /*!
        This class defines an interface for gSparse's graph reading functionality.
        New input source should derive from this class. 
    */
    class IGraphReader
	{
	public:
        //!  Default Constructor for IGraphReader.
		IGraphReader() = default;
        //! A pure virtual member to read Graph Data by passing two reference for input.
        /*!
        \param Edges: An Eigen Matrix to receive the Edge list.
        \param Weights: An Eigen Matrix to receive the Weight list associated to the Edge list.
        */
		virtual void Read(gSparse::EdgeMatrix & Edges,
			gSparse::PrecisionRowMatrix & Weights) = 0;
        //!  Default virtual destructor
		virtual ~IGraphReader() = default;
	};
    typedef std::shared_ptr<gSparse::IGraphReader> GraphReader; 
}
#endif