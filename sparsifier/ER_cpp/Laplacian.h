#ifndef _LAPLACIAN_H_
#define _LAPLACIAN_H_

#include "Matrix.h"
#include "macros.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

LLmatp_t LLmatp(SparseMatrixCSC &a);
LDLinv_t approxchol(LLmatp_t &a);
ApproxCholPQ_t ApproxCholPQ(vector<int> &degs);
void forwardSubstitution(LDLinv_t &ldli, vector<double> &y);
void backwardSubstitution(LDLinv_t &ldli, vector<double> &y);
vector<double> LDLSolver(LDLinv_t &ldli, vector<double> &b);
SparseMatrixCSC lap(SparseMatrixCSC &a);
vector<double> pcg(SparseMatrixCSC &mat, vector<double> b, LDLinv_t &ldli,
                   bool verbose, double tol, int maxits, int stag_test);

#ifdef READ_LA
vector<vector<double>> approxchol_lapGreedy(SparseMatrixCSC &a,
                                            SparseMatrixCSC &la,
                                            vector<vector<double>> &bs);
#else
vector<vector<double>> approxchol_lapGreedy(SparseMatrixCSC &a,
                                            vector<vector<double>> &bs);
#endif

int keyMap(int x, int n);
double dot(vector<double> &a, vector<double> &b);
double norm(vector<double> &a);
// y = a * x + y
void axpy2(double a, vector<double> &x, vector<double> &y);
// p = z + beta * p
void bzbeta(double beta, vector<double> &p, vector<double> &z);

#endif
