// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#ifndef GSPARSE_ER_POLICY_EXACTERJACOBICG_HPP
#define GSPARSE_ER_POLICY_EXACTERJACOBICG_HPP

#include "../../Config.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace gSparse {
namespace ER {
namespace Policy {
/// \ingroup EffectiveResistance
///
/// This class implements Spectral Sparsifier by Effective Weight Sampling.
/// Adaptation from http://ccom.uprrp.edu/~ikoutis/SpectralAlgorithms.htm
/// The algorithm leverages Conjugated Graident with Jacobi preconditioner to
/// solve linear system
///
class ExactERJacobiCG {
protected:
  /// This function calculates Effective Resistance and return computation
  /// status. \param er A row matrix to receive the EffectiveResistance value
  /// \param graph A std::shared_ptr<IGraph> object representing the graph to
  /// calculate resistance \param eps Error tolerance for conjugated gradient.
  /// Default is 1.0f. \param JLTol Tolerance for JL projection Matrix. Default
  /// is 0.5f. (See http://ccom.uprrp.edu/~ikoutis/SpectralAlgorithms.htm.)
  /// \param maxIter  Maximum iteration for conjugated gradient. Default is 300
  /// iterations.
  inline gSparse::COMPUTE_INFO _calculateER(gSparse::PrecisionRowMatrix &er,
                                            const gSparse::Graph &graph,
                                            int maxIter = 300) {
    er = gSparse::PrecisionRowMatrix::Zero(graph->GetEdgeCount(), 1);
    for (int i = 0; i != graph->GetEdgeCount(); ++i) {
      Eigen::ConjugateGradient<gSparse::SparsePrecisionMatrix,
                               Eigen::Lower | Eigen::Upper>
          cg;
      cg.setMaxIterations(maxIter);
      auto b = graph->GetIncidentMatrix().row(i);
      cg.compute(graph->GetLaplacianMatrix());
      Eigen::VectorXd x = cg.solve(b.transpose());
      b.toDense();
      er(i) = b.toDense() * x;
    }
    // Non finite number goes to zero
    er = er.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.0; });
    return gSparse::SUCCESSFUL;
  }
};
} // namespace Policy

} // namespace ER
} // namespace gSparse
#endif
