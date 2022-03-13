
// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef GSPARSE_ER_APPROXIMATEER_HPP
#define GSPARSE_ER_APPROXIMATEER_HPP

#include "../Interface/EffectiveResistance.hpp"
#include "../Config.hpp"
#include "../Util/JL.hpp"  // Building Random Projection

// Approximate ER Policies
#include "Policy/AproxERSLMJacobiCG.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
namespace gSparse 
{
    namespace ER 
    {

        /// \ingroup EffectiveResistance
        ///
        /// This class calculates Approxmation of Graph's Effective Resistance.
        /// There are many approach to this problem. Such approach is set via class Policy.
        template <typename Policy> 
        class _ApproximateER : public IEffectiveResistance, private Policy
        {
        public:
            using Policy::_calculateER;
            inline gSparse::COMPUTE_INFO CalculateER( gSparse::PrecisionRowMatrix & er,
                const gSparse::Graph & graph)
            {
                return _calculateER(er, graph);
            }
        };
        typedef _ApproximateER<Policy::AproxERSLMJacobiCG> ApproximateER; //<! ApproximateER class to calculate effective resistance
    }
}
#endif

