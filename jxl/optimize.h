// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Utility functions for optimizing multi-dimensional nonlinear functions.

#ifndef JXL_OPTIMIZE_H_
#define JXL_OPTIMIZE_H_

#include <stdio.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace jxl {
namespace optimize {

// Runs Nelder-Mead like optimization. Runs for max_iterations times,
// fun gets called with a vector of size dim as argument, and returns the score
// based on those parameters (lower is better). Returns a vector of dim+1
// dimensions, where the first value is the optimal value of the function and
// the rest is the argmin value. Use init to pass an initial guess or where
// the optimal value is.
//
// Usage example:
//
// RunSimplex(2, 0.1, 100, [](const vector<float>& v) {
//   return (v[0] - 5) * (v[0] - 5) + (v[1] - 7) * (v[1] - 7);
// });
//
// Returns (0.0, 5, 7)
std::vector<double> RunSimplex(
    int dim, double amount, int max_iterations,
    const std::function<double((const std::vector<double>&))>& fun);
std::vector<double> RunSimplex(
    int dim, double amount, int max_iterations, const std::vector<double>& init,
    const std::function<double((const std::vector<double>&))>& fun);

template <typename T>
std::vector<T> operator+(const std::vector<T>& x, const std::vector<T>& y) {
  std::vector<T> z(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    z[i] = x[i] + y[i];
  }
  return z;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& x, const std::vector<T>& y) {
  std::vector<T> z(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    z[i] = x[i] - y[i];
  }
  return z;
}

template <typename T>
std::vector<T> operator*(T v, const std::vector<T>& x) {
  std::vector<T> y(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    y[i] = v * x[i];
  }
  return y;
}

template <typename T>
T operator*(const std::vector<T>& x, const std::vector<T>& y) {
  T r = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    r += x[i] * y[i];
  }
  return r;
}

// Implementation of the Scaled Conjugate Gradient method described in the
// following paper:
//   Moller, M. "A Scaled Conjugate Gradient Algorithm for Fast Supervised
//   Learning", Neural Networks, Vol. 6. pp. 525-533, 1993
//   http://sci2s.ugr.es/keel/pdf/algorithm/articulo/moller1990.pdf
//
// The Function template parameter is a class that has the following method:
//
//   // Returns the value of the function at point w and sets *df to be the
//   // negative gradient vector of the function at point w.
//   double Compute(const vector<double>& w, vector<double>* df) const;
//
// Returns a vector w, such that |df(w)| < grad_norm_threshold.
template <typename T, typename Function>
std::vector<T> OptimizeWithScaledConjugateGradientMethod(
    const Function& f, const std::vector<T>& w0, const T grad_norm_threshold,
    int max_iters) {
  const size_t n = w0.size();
  const T rsq_threshold = grad_norm_threshold * grad_norm_threshold;
  const T sigma0 = static_cast<T>(0.0001);
  T lambda = static_cast<T>(0.000001);
  std::vector<T> w(w0);
  std::vector<T> p(n);
  std::vector<T> r(n);
  T fw = f.Compute(w, &r);
  T rsq = r * r;
  T psq = rsq;
  T mu = rsq;
  p = r;
  for (int k = 1; rsq > rsq_threshold; ++k) {
    if (max_iters > 0 && k > max_iters) break;
    T sigma = sigma0 / std::sqrt(psq);
    std::vector<T> r2(n);
    std::vector<T> w2 = w + (sigma * p);
    f.Compute(w2, &r2);
    T delta = (mu - (p * r2)) / sigma;
    T delta1 = delta + lambda * psq;

    if (delta1 <= 0) {
      lambda = -2.0 * delta / psq;
      delta1 = delta + lambda * psq;
    }

    bool success = true;
    T alpha;
    T fw1;
    T Delta;
    std::vector<T> w1(n);
    std::vector<T> r1(n);

    do {
      alpha = mu / delta1;
      w1 = w + (alpha * p);
      fw1 = f.Compute(w1, &r1);
      const T div = mu * alpha;
      Delta = div == 0 ? 0 : 2 * (fw - fw1) / div;
      success = (fw1 <= fw);
      if (!success) {
        lambda += delta1 * (1 - Delta) / psq;
        delta1 = delta + lambda * psq;
      }
    } while (!success);

    T r1sq = r1 * r1;
    T beta = k % n == 0 ? 0.0 : (r1sq - (r1 * r)) / mu;

#if SCG_DEBUG
    printf(
        "Step %3d fw=%10.2f |dfw|=%7.3f |p|=%6.2f "
        "delta=%9.6f lambda=%6.4f mu=%8.4f alpha=%8.4f "
        "beta=%5.3f Delta=%5.3f\n",
        k, fw, sqrt(rsq), sqrt(psq), delta, lambda, mu, alpha, beta, Delta);
#endif

    if (Delta >= 0.75) {
      lambda *= 0.25;
    } else if (Delta < 0.25) {
      lambda += delta1 * (1 - Delta) / psq;
    }

    w = w1;
    fw = fw1;
    r = r1;
    rsq = r1sq;
    p = r + (beta * p);
    psq = p * p;
    mu = p * r;
  }
  return w;
}

}  // namespace optimize
}  // namespace jxl

#endif  // JXL_OPTIMIZE_H_
