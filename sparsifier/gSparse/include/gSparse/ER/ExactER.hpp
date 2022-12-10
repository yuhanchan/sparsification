// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#ifndef GSPARSE_ER_EXACTER_HPP
#define GSPARSE_ER_EXACTER_HPP

#include "../Config.hpp"

#include "../Interface/EffectiveResistance.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Policy/ExactERJacobiCG.hpp"
#include <iostream>
namespace gSparse {
namespace ER {
/// \ingroup EffectiveResistance
///
/// This class calculates Exact value of Graph's Effective Resistance.
/// There are many approach to this problem. Such approach is set via class
/// Policy. Warning: The algorithm for calculating Exact Resistance is very
/// slow.
template <typename Policy>
class _ExactER : public IEffectiveResistance, private Policy {
public:
  using Policy::_calculateER;
  inline gSparse::COMPUTE_INFO CalculateER(gSparse::PrecisionRowMatrix &er,
                                           const gSparse::Graph &graph) {
    return _calculateER(er, graph);
  }
};
typedef _ExactER<Policy::ExactERJacobiCG>
    ExactER; //<! ExactER class to calculate effective resistance
} // namespace ER
} // namespace gSparse
#endif
