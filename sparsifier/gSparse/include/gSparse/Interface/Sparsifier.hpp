// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#ifndef GSPARSE_INTERFACE_SPARSIFIER_HPP
#define GSPARSE_INTERFACE_SPARSIFIER_HPP

#include "../Config.hpp" // Library configuration
#include "Graph.hpp"
#include <Eigen/Dense> // Eigen Matrix
#include <cstddef>     // size_t
#include <memory>      //shared_ptr

namespace gSparse {
//!  An interface class for Graph Sparsifier
/*!
    This class defines an interface for gSparse's Sparsifier
*/
class ISparsifier {
public:
  //! A pure virtual member to computer sparsifier weight.
  virtual COMPUTE_INFO Compute() = 0;
  //! A pure virtual member to get computation status of Sparsifier.
  virtual COMPUTE_INFO GetInfo() const = 0;
  //! A pure virtual member to sparsifier a graph
  virtual gSparse::Graph GetSparsifiedGraph() = 0;
  virtual ~ISparsifier() = default;

protected:
  ISparsifier() = default;
};
typedef std::shared_ptr<ISparsifier> Sparsifier;
} // namespace gSparse
#endif
