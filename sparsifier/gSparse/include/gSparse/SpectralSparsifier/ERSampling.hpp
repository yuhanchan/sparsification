// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef GSPARSE_SPECTRALSPARSIFIER_EFFECTIVERESISTANCESAMPLING_HPP
#define GSPARSE_SPECTRALSPARSIFIER_EFFECTIVERESISTANCESAMPLING_HPP

// Internal includes
#include "../Config.hpp"
#include "../Interface/Sparsifier.hpp"
#include "../Util/Sampling.hpp"

// ER Policies
#include "../ER/ApproximateER.hpp"
#include "../ER/ExactER.hpp"

#include <random>            // distributions
#include <vector>            // Vector
#include <unordered_map>     // Hashtable
namespace gSparse
{
    namespace SpectralSparsifier 
    {
        /// \ingroup SpectralSparsifier
        ///
        /// There are many different approaches to calculate Effective Resistance of a graph.
        /// This is enum type defines the method used by ERSampling class
        ///
        enum ER_METHODS
        {
            APPROXIMATE_ER = 0,
            EXACT_ER = 1
        };

        /// \ingroup SpectralSparsifier
        ///
        /// This class implements Spectral Sparsifier by Effective Weight Sampling.
        /// Original paper <https://arxiv.org/pdf/0803.0929.pdf>
        ///
        class ERSampling : public ISparsifier
        {
        protected:
            double _c;                                      //!< C-hyper parameter.
            double _eps;                                    //!< epsilon hyper-parameter.
            gSparse::COMPUTE_INFO _computeInfo;             //!< Status of sparsifier
            gSparse::Graph _graph;                          //!< Graph to Sparsify
            gSparse::PrecisionRowMatrix _er;                //!< Effective Resistance
            gSparse::EffectiveResistance _erCalculator;     //!< Pointer to EffectiveResistance module
                          
            gSparse::SpectralSparsifier::ER_METHODS _erPolicy; //!< EffectiveResistance Calculation Policy
        public:
            ///
            /// Constructor to create a ISparsifier object to perform Spectral Sparsification by Effective Resistance
            ///
            /// \param graph    Smart Pointer to the IGraph object. This is the dense graph for the sparsifier. 
            /// \param C        C hyper-parameter of Spectral Sparsifier by Effective Resistance. 
            ///                 As described by paper. This should be a large constant.
            ///                 Default value is 4.0.
            /// \param Epsilon  Epsilon hyper-parameter. This should be between 0.0 and 1.0
            ///                 The sparsifier will generate (m log m ) / (Epsilon^2) edges. 
            ///                 Thus, lower epsilon will yeild denser sparsifier. High epsilon could yeild a disconnect graph.
            /// \param ERPolicy EffectiveResistance calculation method. 
            ///                 APPROXIMATE_ER:  Default. Approximate EffectiveResistance with an error. This should be sufficient for Spectral Sparsifier.                        
            ///                 EXACT_ER:  Calculate graph exact EffectiveResistance. This is very slow and not recommended.
            ERSampling(const gSparse::Graph & graph, 
            double C = 4.0f, 
            double Epsilon = 0.3f,
            gSparse::SpectralSparsifier::ER_METHODS ERPolicy = gSparse::SpectralSparsifier::APPROXIMATE_ER
            )
            {
                SetC(C);
                SetEpsilon(Epsilon);
                _graph = graph;
                _computeInfo = gSparse::NOT_COMPUTED;
                SetERPolicy(ERPolicy);
            }

            ///
            /// Calculate Effective Resistance of the graph specified in the constructor.
            ///
            virtual inline gSparse::COMPUTE_INFO Compute()
            {
                //Calculate Effective Resistance
                _computeInfo = _erCalculator->CalculateER(_er, _graph);
                return _computeInfo;
            }
            ///
            /// Sampling a sparsifier from a dense graph based on calculate effective weight.
            /// As this is a random algorithm, it may need several attempts to get an acceptable graph.
            ///
            virtual inline gSparse::Graph GetSparsifiedGraph()
            {
                if (_computeInfo == gSparse::NOT_COMPUTED)
                {
                    throw std::logic_error("SpectralSparsifier by ER: User must run Compute before GetSparsifiedGraph()");
                }
                // Build probability distribution
                std::vector<double> samplingWeights;
                samplingWeights.reserve(_er.rows());

                //Generate sampling weights (p) for each edges
                for (std::size_t i = 0; i != _er.rows(); ++i)
                {
                    double temp = _er(i, 0) * _graph->GetWeightList()(i) * _c * std::log(_graph->GetNodeCount()) / std::pow(_eps, 2);
                    double prob = 1.0f < temp ? 1.0f : temp;
                    samplingWeights.push_back(prob);
                }

                // Build a discrete distribution based on sampling weight
                std::discrete_distribution<> samplingDistribution(samplingWeights.begin(), samplingWeights.end());

                // The algorithm samples O(n log n / ep^2) times edges
                std::size_t samplingCount = static_cast<std::size_t>(
                    std::ceil(_graph->GetNodeCount() * std::log(_graph->GetNodeCount()) / std::pow(_eps, 2)));
                // Sample edges from a distribution, then add to a hash table
                std::unordered_map<std::size_t, gSparse::PRECISION> sparsifiedMap;
                for (std::size_t i = 0; i != samplingCount; ++i)
                {
                    
                    std::size_t edgeIndex = gSparse::Util::sample(samplingDistribution);
                    sparsifiedMap[edgeIndex] += _graph->GetWeightList()(edgeIndex) / samplingWeights[edgeIndex];
                }
                
                // Build Graph object from sparsified information
                gSparse::EdgeMatrix resultEdge(sparsifiedMap.size(), 2);
                gSparse::PrecisionRowMatrix resultWeight(sparsifiedMap.size(), 1);
                std::size_t row = 0;
                for (auto x = sparsifiedMap.begin(); x != sparsifiedMap.end(); ++x)
                {
                    resultEdge(row, 0) = _graph->GetEdgeList()(x->first, 0);
                    resultEdge(row, 1) = _graph->GetEdgeList()(x->first, 1);
                    resultWeight(row, 0) = x->second;
                    ++row;
                }
                return std::make_shared<gSparse::UndirectedGraph>(resultEdge, resultWeight);
            }
            ///
            /// Set EffectiveResistance calculation methid.
            ///
            /// \param ERPolicy EffectiveResistance calculation method. 
            ///                 APPROXIMATE_ER:  Default. Approximate EffectiveResistance with an error. This should be sufficient for Spectral Sparsifier.                        
            ///
            inline void SetERPolicy(ER_METHODS policy)
            {
                assert (policy == gSparse::SpectralSparsifier::APPROXIMATE_ER || 
                        policy == gSparse::SpectralSparsifier::EXACT_ER);
                
                switch (policy)
                {
                case gSparse::SpectralSparsifier::EXACT_ER:
                    _erCalculator = std::make_shared<gSparse::ER::ExactER>(); 
                    _erPolicy = gSparse::SpectralSparsifier::EXACT_ER;
                 break;
                case gSparse::SpectralSparsifier::APPROXIMATE_ER:
                default:
                    _erCalculator = std::make_shared<gSparse::ER::ApproximateER>();
                    _erPolicy = gSparse::SpectralSparsifier::APPROXIMATE_ER;
                }
            }
            /// Set Hyper-parameter C
            /// \param C        C hyper-parameter of Spectral Sparsifier by Effective Resistance. 
            ///                 As described by paper. This should be a large constant.
            inline void SetC(double C)
            {
                #ifndef NDEBUG
                    assert(C > 0.0f);
                #endif
                _c = C;
            }
            /// Set Hyper-parameter Epsilon
            /// \param Epsilon  Epsilon hyper-parameter. This should be between 0.0 and 1.0
            ///                 The sparsifier will generate (m log m ) / (Epsilon^2) edges. 
            ///                 Thus, lower epsilon will yeild denser sparsifier. High epsilon could yeild a disconnect graph.
            inline void SetEpsilon(double Epsilon)
            {
                #ifndef NDEBUG
                    assert(Epsilon > 0.0f);
                #endif
                _eps = Epsilon;
            }
            /// Get the sparsifier's current configuration for hyper-parameter C
            inline double GetC() const { return _c; }
            /// Get the sparsifier's current configuration for hyper-parameter Epsilon
            inline double GetEpsilon() const { return _eps; }
            /// Get the sparsifier's current Effective Resistance policy
            inline gSparse::SpectralSparsifier::ER_METHODS GetERPolicy() const { return _erPolicy; }
            /// Get the sparsifier's current computation information
            inline gSparse::COMPUTE_INFO GetInfo() const { return _computeInfo; }
            /// Get the effective resistance of the graph specified at construction
            inline const gSparse::PrecisionRowMatrix & GetEffectiveResistance() const
            {
                return _er;
            }
        };

    }
}
#endif