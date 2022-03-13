
// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef GSPARSE_ER_POLICY_APROXERSLMJACOBICG_HPP
#define GSPARSE_ER_POLICY_APROXERSLMJACOBICG_HPP

#include "../../Config.hpp"
#include "../../Util/JL.hpp"  // Building Random Projection

#include <Eigen/Dense>
#include <Eigen/Sparse>
namespace gSparse 
{
    namespace ER 
    {
        namespace Policy
        {
            /// \ingroup EffectiveResistance
            ///
            /// This class Approximate Effective Weight Resistance
            /// Adaptation from http://ccom.uprrp.edu/~ikoutis/SpectralAlgorithms.htm.
            /// The algorithm leverages Conjugated Graident with Jacobi preconditioner to solve linear system
            ///
            class AproxERSLMJacobiCG
            {
            protected:
                /// This function calculates Effective Resistance and return computation status.
                /// \param er A row matrix to receive the EffectiveResistance value
                /// \param graph A std::shared_ptr<IGraph> object representing the graph to calculate resistance
                /// \param eps Error tolerance for conjugated gradient. Default is 1.0f.
                /// \param JLTol Tolerance for JL projection Matrix. Default is 0.5f. (See http://ccom.uprrp.edu/~ikoutis/SpectralAlgorithms.htm.)
                /// \param maxIter  Maximum iteration for conjugated gradient. Default is 300 iterations.
                inline gSparse::COMPUTE_INFO _calculateER(
                    gSparse::PrecisionRowMatrix & er,
                    const gSparse::Graph & graph,
                    double eps = 1.0f,
                    double JLTol = 0.5,
                    int maxIter = 300
                    )
                {
                    er = gSparse::PrecisionRowMatrix::Zero(graph->GetEdgeCount(), 1);

                    std::size_t scale = static_cast<size_t>(
                                std::ceil(
                                std::log2(
                                static_cast<double>(graph->GetIncidentMatrix().cols()) / eps)));

                    for (int i = 1; i != scale + 1; ++i)
                    {
                        Eigen::VectorXd x;
                        gSparse::PrecisionMatrix Q =
                        gSparse::Util::randomProjectionMatrix(1, 
                                                                graph->GetIncidentMatrix().rows(), 
                                                                static_cast<double>(scale), 
                                                                JLTol);

                        gSparse::PrecisionMatrix Y = (Q * graph->GetWeightMatrix().cwiseSqrt() * graph->GetIncidentMatrix());

                        // solve Linear system with 300 max iteration
                        Eigen::ConjugateGradient<gSparse::SparsePrecisionMatrix, Eigen::Lower | Eigen::Upper  > cg;
                        cg.setMaxIterations(maxIter);
                        x = cg.compute(graph->GetLaplacianMatrix()).solve(Y.transpose());
                                    
                        if (cg.info() != Eigen::Success)
                        {
                            // Does not converge this iteration. Keeps going.
                            continue;
                        }
                        for (std::size_t j = 0; j != graph->GetEdgeCount(); ++j)
                        {
                            er(j) += pow(std::abs(x(graph->GetEdgeList()(j, 0)) - x(graph->GetEdgeList()(j, 1))), 2.0f);
                        }
                    }
                    // Non finite element goes to zero
                    er = er.unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });
                    if (er.rows() != graph->GetEdgeCount())
                        return gSparse::NOT_CONVERGING;
                    return gSparse::SUCCESSFUL;       
                }
            };
        }
    }
}
#endif

