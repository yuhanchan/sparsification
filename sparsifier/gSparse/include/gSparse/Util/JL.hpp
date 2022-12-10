// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#ifndef GSPARSE_UTIL_JL_HPP
#define GSPARSE_UTIL_JL_HPP

#include "../Config.hpp"
#include <Eigen/Dense>

namespace gSparse {
namespace Util {
//! randomProjectionMatrix creates a Johnson-Lindenstrauss lemma projection
//! matrix.
/*!
\param rows: Number of rows for generated matrix.
\param cols: Number of columns for generated matrix.
\param scale: square root of Scale will divide the value of the JL Matrix.
Default is 1.0. \param tolProb: Tolerance threshold value between 0.0 and 1.0.
Higher tolerance means less likely to get positive matrix. Default is 0.5.
*/
inline gSparse::PrecisionMatrix randomProjectionMatrix(std::size_t rows,
                                                       std::size_t cols,
                                                       double scale = 1.0f,
                                                       double tolProb = 0.5f) {
#ifndef NDEBUG
  assert(tolProb <= 1.0f);
  assert(tolProb >= 0.0f);
  assert(scale != 0.0f);
  assert(rows > 0);
  assert(cols > 0);
#endif

  gSparse::PrecisionMatrix result =
      (gSparse::PrecisionMatrix::Random(rows, cols).array() + 1.0) / 2.0;

  result = result.unaryExpr([=](gSparse::PRECISION x) {
    return x > tolProb ? 1.0 / sqrt(scale) : -1.0 / std::sqrt(scale);
  });
  // Copy elision will optimize return by value.
  return result;
}
} // namespace Util
} // namespace gSparse

#endif
