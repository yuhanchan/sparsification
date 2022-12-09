#include <experimental/algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <set>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <utility>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "timer.h"

#define parallel_for _Pragma("omp parallel for") for

#define SEED 6

typedef std::pair<int, int> edge_t;

std::mutex mtx;

void random_pruning(Graph *g, int64_t num_edges, int64_t num_edges_to_prune) {
  std::cout << "Doing random pruning...\n";

  srand(SEED);

  std::vector<int64_t> edges_to_prune;
  edges_to_prune.reserve(num_edges);
  for (int64_t i = 0; i < num_edges; i++) {
    edges_to_prune.push_back(i);
  }
  std::random_shuffle(edges_to_prune.begin(), edges_to_prune.end());

  for (int64_t i = 0; i < num_edges_to_prune; i++) {
    // std::cout<<"Pruning edge "<<edges_to_prune[i]<<"\n";
    g->SetIthIndex(edges_to_prune[i]);
  }

  // std::set<int64_t> prune_indices;
  // while((int64_t)prune_indices.size() < num_edges_to_prune) {
  // int64_t ind = rand() % (num_edges);
  // prune_indices.insert(ind);
  // }

  // for(auto it : prune_indices)
  // g->SetIthIndex(it);
}

int64_t out_threshold_pruning(Graph *g, int64_t num_edges,
                              int64_t outgoing_edges_threshold) {
  int64_t num_pruned_edges = 0;

  std::cout << "Threshold pruning based on outgoing edges...\n";
  std::cout << "Pruning threshold: " << outgoing_edges_threshold << std::endl;

  Timer t_prune;
  t_prune.Start();
  for (int64_t i = 0; i < g->num_nodes(); ++i) {
    // printf("Node %d has out_degree: %d\n", i, g->out_degree(i));
    if (g->out_degree(i) > outgoing_edges_threshold) {
      int64_t num_edges_to_prune = g->out_degree(i) - outgoing_edges_threshold;
      srand(SEED);
      std::set<int64_t> prune_indices;
      while ((int64_t)prune_indices.size() < num_edges_to_prune) {
        int64_t ind = rand() % (g->out_degree(i));
        prune_indices.insert(ind);
      }
      num_pruned_edges += prune_indices.size();
      for (auto it : prune_indices) {
        g->SetIthNeighborID(i, it);
      }
    }
  }
  t_prune.Stop();
  PrintStep("[TimingStat] Time to prune (s):", t_prune.Seconds());

  return num_pruned_edges;
}

int64_t in_threshold_pruning(Graph *g, int64_t num_edges,
                             int64_t incoming_edges_threshold) {
  int64_t num_pruned_edges = 0;

  std::cout << "Threshold pruning based on incoming edges...\n";
  std::cout << "Pruning threshold: " << incoming_edges_threshold << std::endl;

  Timer t_prune;
  t_prune.Start();
  for (int64_t i = 0; i < g->num_nodes(); ++i) {
    // printf("Node %d has in_degree: %d\n", i, g->in_degree(i));
    if (g->in_degree(i) > incoming_edges_threshold) {
      int64_t num_edges_to_prune = g->in_degree(i) - incoming_edges_threshold;
      srand(SEED);
      std::set<int64_t> prune_indices;
      while ((int64_t)prune_indices.size() < num_edges_to_prune) {
        int64_t ind = rand() % (g->in_degree(i));
        prune_indices.insert(ind);
      }
      num_pruned_edges += prune_indices.size();
      for (auto it : prune_indices) {
        int64_t idx_of_dest = g->find_idx_of_dest(*(g->in_index_[i] + it), i);
        assert(idx_of_dest != -1);
        g->SetIthNeighborID(*(g->in_index_[i] + it), (int64_t)idx_of_dest);
      }
    }
  }
  t_prune.Stop();
  PrintStep("[TimingStat] Time to prune (s):", t_prune.Seconds());

  return num_pruned_edges;
}

int64_t sym_threshold_pruning(Graph *g, int64_t num_edges, int64_t threshold) {
  int64_t num_pruned_edges = 0;

  std::cout << "Threshold pruning based on symmetrical edges...\n";
  std::cout << "Pruning threshold: " << threshold << std::endl;

  Timer t_prune;
  t_prune.Start();
  for (int64_t i = 0; i < g->num_nodes(); ++i) {
    if (g->out_degree(i) > threshold) {
      int64_t num_edges_to_prune = g->out_degree(i) - threshold;
      srand(SEED);
      std::set<int64_t> prune_indices;
      while ((int64_t)prune_indices.size() < num_edges_to_prune) {
        int64_t ind = rand() % (g->out_degree(i));
        prune_indices.insert(ind);
      }
      num_pruned_edges += prune_indices.size();
      for (auto it : prune_indices) {
        g->SetIthNeighborID(i, it);
      }
    }
  }
  t_prune.Stop();
  PrintStep("[TimingStat] Time to prune (s):", t_prune.Seconds());

  return num_pruned_edges;
}

int write_el_to_file(const Graph *g, std::string pruned_graph_el_filename,
                     bool post_symmetrize) {
  std::ofstream pruned_graph_file(pruned_graph_el_filename);
  std::set<edge_t> edges;

  if (post_symmetrize) {
    for (NodeID i = 0; i < g->num_nodes(); ++i) {
      for (auto it : g->out_neigh(i)) {
        if (it > 0) { // Pruned neighbor ID is set to -1
          edges.insert(std::make_pair(i, it));
          edges.insert(std::make_pair(it, i));
        }
      }
    }
    for (auto it : edges) {
      pruned_graph_file << it.first << " " << it.second << "\n";
    }
  } else {
    for (NodeID i = 0; i < g->num_nodes(); ++i) {
      for (auto it : g->out_neigh(i)) {
        if (it > 0) { // Pruned neighbor ID is set to -1
          edges.insert(std::make_pair(i, it));
        }
      }
    }
    for (auto it : edges) {
      pruned_graph_file << it.first << " " << it.second << "\n";
    }
  }
  pruned_graph_file.close();
  return edges.size();
}

void write_edge_index_to_file(const Graph *g,
                              std::string pruned_graph_edge_index_filename,
                              int64_t num_pruned_edges) {
  std::ofstream pruned_graph_file(pruned_graph_edge_index_filename);
  int64_t *pruned_edge_indices = g->GetPrunedEdgeIDs(num_pruned_edges);
  for (int64_t i = 0; i < num_pruned_edges; ++i) {
    pruned_graph_file << pruned_edge_indices[i] << "\n";
  }
  pruned_graph_file.close();
  delete[] pruned_edge_indices;
}

int get_num_nodes_with_in_degree_more_than(const Graph *g, int thres) {
  int num_nodes = 0;
  for (int i = 0; i < g->num_nodes(); ++i) {
    if (g->in_degree(i) > thres)
      num_nodes++;
  }
  return num_nodes;
}

int main(int argc, char *argv[]) {

  CLApp cli(argc, argv, "graph-pruning");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();

  // std::cout<<"Number of nodes with degree > 0:
  // "<<get_num_nodes_with_in_degree_more_than(&g, 0)<<std::endl;
  // return 0;

  Timer t_overall;
  t_overall.Start();
  int64_t num_nodes = g.num_nodes();
  int64_t num_edges = g.num_edges();
  float pruning_level = cli.get_pruning_level();
  std::string pruning_type = cli.get_pruning_type();
  std::string pruned_graph_file = cli.get_pruned_graph_filename();
  int64_t pruning_threshold = cli.get_pruning_threshold();
  int64_t num_edges_to_prune = pruning_level * num_edges;
  bool post_symmetrize = cli.post_symmetrize();

  std::cout << "-------- Graph stats ---------\n";
  std::cout << "Num nodes: " << num_nodes << ", num edges: " << num_edges
            << std::endl;
  std::cout << "Pruning level: " << pruning_level << std::endl;
  std::cout << "Max. num edges for threshold pruning: " << pruning_threshold
            << std::endl;
  std::cout << "Pruning type: " << pruning_type << std::endl;
  std::cout << "File name of the output pruned graph edgelist: "
            << pruned_graph_file << std::endl;
  std::cout << "Num edges to prune: " << num_edges_to_prune << std::endl;

  // g.PrintEdges();

  std::cout << "--- Pruning edges ---\n";
  if (pruning_type == "random") {
    // Random pruning
    random_pruning(&g, num_edges, num_edges_to_prune);
  } else if (pruning_type == "out_threshold") {
    // Threshold pruning based on outgoing edges
    num_edges_to_prune =
        out_threshold_pruning(&g, num_edges, pruning_threshold);
    // TODO:
    // Update num_neighbors (after pruning) and
    // make sure that the number if <= threshold
  } else if (pruning_type == "in_threshold") {
    // Threshold pruning based on incoming edges
    num_edges_to_prune = in_threshold_pruning(&g, num_edges, pruning_threshold);
  } else {
    std::cout << "[ERROR] Unknown pruning method, "
              << "double check the pruning type flag '-q'!\n";
    return -1;
  }

  // Print a pruned graph to a file
  int num_edges_after_pruning =
      write_el_to_file(&g, pruned_graph_file, post_symmetrize);
  // write_edge_index_to_file(&g, pruned_graph_file, num_edges_to_prune);
  t_overall.Stop();
  PrintStep("[TimingStat] Time to prune and write to file (s):",
            t_overall.Seconds());
  if (pruning_type == "random") {
    std::cout << "Target pruning level: " << pruning_level
              << " . Actual pruning level: "
              << 1.0 - (num_edges_after_pruning * 1.0 / num_edges) << std::endl;
  } else {
    std::cout << "pruning threshold: " << pruning_threshold
              << " . Actual pruning level: "
              << 1.0 - (num_edges_after_pruning * 1.0 / num_edges) << std::endl;
  }

  return 0;
}
