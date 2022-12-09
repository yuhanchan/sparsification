// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

#include <time.h>

/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. It perform
updates in the pull direction to remove the need for atomics, and it allows
new values to be immediately visible (like Gauss-Seidel method). The prior PR
implemention is still available in src/pr_spmv.cc.
*/


using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;


pvector<ScoreT> PageRankPullGS(const Graph &g, int max_iters,
                             double epsilon = 0) {
  clock_t start = clock();
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> outgoing_contrib(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    outgoing_contrib[n] = init_score / g.out_degree(n);
  for (int iter=0; iter < max_iters; iter++) {
    double error = 0;
    #pragma omp parallel for reduction(+ : error) schedule(dynamic, 16384)
    for (NodeID u=0; u < g.num_nodes(); u++) {
      ScoreT incoming_total = 0;
      for (NodeID v : g.in_neigh(u))
        incoming_total += outgoing_contrib[v];
      ScoreT old_score = scores[u];
      scores[u] = base_score + kDamp * incoming_total;
      error += fabs(scores[u] - old_score);
      outgoing_contrib[u] = scores[u] / g.out_degree(u);
    }
    printf(" %2d    %lf\n", iter, error);
    if (error < epsilon)
      break;
  }
  cout<<"CPU Time: "<<(double)(clock() - start)/CLOCKS_PER_SEC<<" seconds"<<endl;
  return scores;
}

pvector<ScoreT> WeightedPageRankPullGS(const WGraph &g, int max_iters,
                             double epsilon = 0) {
  clock_t start = clock();
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> outgoing_contrib(g.num_nodes());
  float *out_weights_ = g.get_out_weights();
  // // print out the out_weights_
  // for (int i = 0; i < 10; i++) {
  //   if (out_weights_[i] != 0) 
  //     cout << "out_weights_[" << i << "] = " << out_weights_[i] << endl;
  // }
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    outgoing_contrib[n] = init_score / out_weights_[n];
  for (int iter=0; iter < max_iters; iter++) {
    double error = 0;
    #pragma omp parallel for reduction(+ : error) schedule(dynamic, 16384)
    for (NodeID u=0; u < g.num_nodes(); u++) {
      ScoreT incoming_total = 0;
      for (WNode v : g.in_neigh(u))
        incoming_total += outgoing_contrib[v.v] * v.w;
      ScoreT old_score = scores[u];
      scores[u] = base_score + kDamp * incoming_total;
      error += fabs(scores[u] - old_score);
      outgoing_contrib[u] = scores[u] / out_weights_[u];
    }
    printf(" %2d    %lf\n", iter, error);
    if (error < epsilon)
      break;
  }
  cout<<"CPU Time: "<<(double)(clock() - start)/CLOCKS_PER_SEC<<" seconds"<<endl;
  return scores;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}

void WeightedPrintTopScores(const WGraph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}

void PrintAllScores(const Graph &g, const pvector<ScoreT> &scores) {
  for (NodeID n=0; n < g.num_nodes(); n++) {
    cout << n << ":" << scores[n] << endl;
  }
}

void PrintTopOnePercentScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, int(g.num_nodes()/100));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}

void WeightedPrintAllScores(const WGraph &g, const pvector<ScoreT> &scores) {
  for (NodeID n=0; n < g.num_nodes(); n++) {
    cout << n << ":" << scores[n] << endl;
  }
}

void WeightedPrintTopOnePercentScores(const WGraph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, int(g.num_nodes()/100));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}

// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool WeightedPRVerifier(const WGraph &g, const pvector<ScoreT> &scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  float *out_weights_ = g.get_out_weights();
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / out_weights_[u];
    for (WNode v : g.out_neigh(u))
      incomming_sums[v.v] += outgoing_contrib * v.w;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}


int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;

  std::string filename_ = cli.filename();
  if (filename_ != "") {
    std::size_t suff_pos = filename_.rfind('.');
    if (suff_pos == std::string::npos) {
      std::cout << "Could't find suffix of " << filename_ << std::endl;
      std::exit(-1);
    }
    if (filename_.substr(suff_pos, 2) == ".w") {
      std::cout << "Weighted PageRank" << std::endl;
      WeightedBuilder b(cli);
      WGraph g = b.MakeGraph();
      auto PRBound = [&cli] (const WGraph &g) {
        return WeightedPageRankPullGS(g, cli.max_iters(), cli.tolerance());
      };
      auto VerifierBound = [&cli] (const WGraph &g, const pvector<ScoreT> &scores) {
        return WeightedPRVerifier(g, scores, cli.tolerance());
      };
      BenchmarkKernel(cli, g, PRBound, WeightedPrintAllScores, VerifierBound);
    } else {
      std::cout<< "Unweighted PageRank"<< std::endl;
      Builder b(cli);
      Graph g = b.MakeGraph();
      auto PRBound = [&cli] (const Graph &g) {
        return PageRankPullGS(g, cli.max_iters(), cli.tolerance());
      };
      auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
        return PRVerifier(g, scores, cli.tolerance());
      };
      BenchmarkKernel(cli, g, PRBound, PrintAllScores, VerifierBound);
    }
  }

  return 0;
}
