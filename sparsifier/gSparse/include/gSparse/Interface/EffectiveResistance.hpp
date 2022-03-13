// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef GSPARSE_INTERFACE_EFFECTIVERESISTANCE_HPP
#define GSPARSE_INTERFACE_EFFECTIVERESISTANCE_HPP

#include <memory>        //shared_ptr
#include "../Config.hpp" // Library configuration

namespace gSparse
{
    //!  An interface class for Effective Resistance Calculator
    /*!
        This class defines an interface for gSparse's EffectiveResistance
    */
	class IEffectiveResistance
	{
	public:
        //! A pure virtual member to computer sparsifier weight.
		virtual gSparse::COMPUTE_INFO CalculateER( gSparse::PrecisionRowMatrix &, const gSparse::Graph & ) = 0;
		virtual ~IEffectiveResistance() = default;
	protected:
		IEffectiveResistance() = default;
	};
	typedef std::shared_ptr<IEffectiveResistance> EffectiveResistance;
}
#endif



