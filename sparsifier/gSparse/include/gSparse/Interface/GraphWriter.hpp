// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#ifndef GSPARSE_INTERFACE_GRAPHWRITER_HPP
#define GSPARSE_INTERFACE_GRAPHWRITER_HPP

#include <Eigen/Dense> // Eigen Matrix
#include <cstddef>     // size_t
#include <memory>      //shared_ptr

#include "../Config.hpp" // Library configuration
#include "Graph.hpp"     // Graph Interface

namespace gSparse {
//!  An interface class for Graph Output.
/*!
    This class defines an interface for gSparse's graph writing functionality.
    New output source should derive this function.
*/
class IGraphWriter {
public:
  //!  Default Constructor for IGraphWriter.
  IGraphWriter() = default;

  //! A pure virtual member write Graph Data by passing an IGraph object.
  /*!
  \param graph: A shared_ptr of an IGraph Object
  */
  virtual void Write(const gSparse::Graph &graph) = 0;

  //! A pure virtual member write Graph Data by passing Edge List.
  /*!
  \param Edges: An Eigen Matrix representings Graph Edges.
  */
  virtual void Write(const gSparse::EdgeMatrix &Edges) = 0;

  //! A pure virtual member write read Graph Data by passing an Edge and a
  //! Weight Lists.
  /*!
  \param Edges: An Eigen Matrix of Graph's Edges.
  \param Weights: An Eigen Matrix of Graph's Weight.
  */
  virtual void Write(const gSparse::EdgeMatrix &Edges,
                     const gSparse::PrecisionRowMatrix &Weights) = 0;

  //!  Default Constructor for IGraphWriter.
  virtual ~IGraphWriter() = default;
};
typedef std::shared_ptr<gSparse::IGraphWriter> GraphWriter;
} // namespace gSparse
#endif
