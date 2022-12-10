#include "CmdArgs.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;

typedef pair<int, int> edge_t;
typedef pair<edge_t, float> wedge_t;

void dw_to_udw(ifstream &in, ofstream &out) {
  int u, v;
  float w;
  while (in >> u >> v >> w) {
    if (u < v) {
      out << u << " " << v << " " << w << endl;
    }
  }
}

void udw_to_dw(ifstream &in, ofstream &out) {
  int u, v;
  float w;
  while (in >> u >> v >> w) {
    out << u << " " << v << " " << w << endl;
    out << v << " " << u << " " << w << endl;
  }
}

void duw_to_uduw(ifstream &in, ofstream &out) {
  int u, v;
  while (in >> u >> v) {
    if (u < v) {
      out << u << " " << v << endl;
    }
  }
}

void uduw_to_duw(ifstream &in, ofstream &out) {
  int u, v;
  while (in >> u >> v) {
    out << u << " " << v << endl;
    out << v << " " << u << endl;
  }
}

void unweight(ifstream &in, ofstream &out) {
  int u, v;
  float w;
  while (in >> u >> v >> w) {
    out << u << " " << v << endl;
  }
}

/* symmetrize based on upper triangle */
void symmetrize(ifstream &in, ofstream &out) {
  string first_line;
  getline(in, first_line);
  stringstream ss(first_line);
  int num_count_per_line = 0;
  string num;
  while (getline(ss, num, ' ')) {
    cout << num << endl;
    num_count_per_line++;
  }

  // reset cursor
  in.clear();
  in.seekg(0, ios::beg);

  if (num_count_per_line == 2) {
    set<edge_t> edges;
    cout << "Input file has 2 numbers per line, treat as un-weighted." << endl;
    int u, v;
    while (in >> u >> v) {
      edges.insert(edge_t(u, v));
      edges.insert(edge_t(v, u));
    }
    for (auto it : edges) {
      out << it.first << " " << it.second << endl;
    }
  } else if (num_count_per_line == 3) {
    set<wedge_t> edges;
    cout << "Input file has 3 numbers per line, treat as weighted." << endl;
    int u, v;
    float w;
    while (in >> u >> v >> w) {
      edges.insert(wedge_t(edge_t(u, v), w));
      edges.insert(wedge_t(edge_t(v, u), w));
    }
    for (auto it : edges) {
      out << it.first.first << " " << it.first.second << " " << it.second
          << endl;
    }
  }
}

struct Comp_edge {
  bool operator()(const edge_t &a, const edge_t &b) {
    return a.first < b.first || (a.first == b.first && a.second < b.second);
  }
};
struct Comp_wedge {
  bool operator()(const wedge_t &a, const wedge_t &b) {
    return a.first.first < b.first.first ||
           (a.first.first == b.first.first && a.first.second < b.first.second);
  }
};
void sort_by_src_then_dst(ifstream &in, ofstream &out) {
  string first_line;
  getline(in, first_line);
  stringstream ss(first_line);
  int num_count_per_line = 0;
  string num;
  while (getline(ss, num, ' ')) {
    cout << num << endl;
    num_count_per_line++;
  }

  // reset cursor
  in.clear();
  in.seekg(0, ios::beg);

  if (num_count_per_line == 2) {
    cout << "Input file has 2 numbers per line, treat as un-weighted." << endl;
    vector<edge_t> edges;
    int u, v;
    while (in >> u >> v) {
      edges.push_back(make_pair(u, v));
    }
    sort(edges.begin(), edges.end(), Comp_edge());
    for (const edge_t &e : edges) {
      out << e.first << " " << e.second << endl;
    }
  } else if (num_count_per_line == 3) {
    cout << "Input file has 3 numbers per line, treat as weighted." << endl;
    vector<wedge_t> edges;
    int u, v;
    float w;
    while (in >> u >> v >> w) {
      edges.push_back(make_pair(make_pair(u, v), w));
    }
    sort(edges.begin(), edges.end(), Comp_wedge());
    for (const wedge_t &e : edges) {
      out << e.first.first << " " << e.first.second << " " << e.second << endl;
    }
  }
}

void zero_base_to_one_base(ifstream &in, ofstream &out) {
  string first_line;
  getline(in, first_line);
  stringstream ss(first_line);
  int num_count_per_line = 0;
  string num;
  while (getline(ss, num, ' ')) {
    cout << num << endl;
    num_count_per_line++;
  }

  // reset cursor
  in.clear();
  in.seekg(0, ios::beg);

  if (num_count_per_line == 2) {
    cout << "Input file has 2 numbers per line, treat as un-weighted." << endl;
    vector<edge_t> edges;
    int u, v;
    while (in >> u >> v) {
      edges.push_back(make_pair(u, v));
    }
    for (const edge_t &e : edges) {
      out << e.first + 1 << " " << e.second + 1 << endl;
    }
  } else if (num_count_per_line == 3) {
    cout << "Input file has 3 numbers per line, treat as weighted." << endl;
    vector<wedge_t> edges;
    int u, v;
    float w;
    while (in >> u >> v >> w) {
      edges.push_back(make_pair(make_pair(u, v), w));
    }
    for (const wedge_t &e : edges) {
      out << e.first.first + 1 << " " << e.first.second + 1 << e.second << endl;
    }
  }
}

void elim_disconnected_nodes(ifstream &in, ofstream &out,
                             string edge_map_file) {
  cout << "Warn: elim_disconnected_nodes assumes row indices in sorted" << endl;

  string first_line;
  getline(in, first_line);
  stringstream ss(first_line);
  int num_count_per_line = 0;
  string num;
  while (getline(ss, num, ' ')) {
    cout << num << endl;
    num_count_per_line++;
  }

  // reset cursor
  in.clear();
  in.seekg(0, ios::beg);

  vector<int> row_ind;
  vector<int> col_ind;
  vector<float> weight;
  int u, v;
  float w;
  if (num_count_per_line == 2) {
    cout << "Input file has 2 numbers per line, treat as un-weighted." << endl;
    while (in >> u >> v) {
      row_ind.push_back(u);
      col_ind.push_back(v);
    }
  } else if (num_count_per_line == 3) {
    cout << "Input file has 3 numbers per line, treat as weighted." << endl;
    while (in >> u >> v >> w) {
      row_ind.push_back(u);
      col_ind.push_back(v);
      weight.push_back(w);
    }
  }

  vector<int> ind_map(row_ind[row_ind.size() - 1] + 10,
                      0); // this is to map the row indices to the new indices
  for (int i = 0; i < row_ind.size(); i++) {
    ind_map[row_ind[i]] = 1;
    ind_map[col_ind[i]] = 1;
  }
  for (int i = 1; i < ind_map.size(); i++) {
    ind_map[i] += ind_map[i - 1];
  }
  for (int i = 0; i < ind_map.size(); i++) {
    ind_map[i] -= 1;
  }

  for (int i = ind_map.size() - 1; i > 0; i--) {
    if (ind_map[i] == ind_map[i - 1]) {
      ind_map[i] = -1;
    }
  }

  if (edge_map_file.size()) {
    ofstream out_edge_map(edge_map_file);
    for (int i = 0; i < ind_map.size(); i++) {
      out_edge_map << ind_map[i] << endl;
    }
  }

  // output
  if (num_count_per_line == 2) {
    for (int i = 0; i < row_ind.size(); i++) {
      out << ind_map[row_ind[i]] << " " << ind_map[col_ind[i]] << endl;
    }
  } else if (num_count_per_line == 3) {
    for (int i = 0; i < row_ind.size(); i++) {
      out << ind_map[row_ind[i]] << " " << ind_map[col_ind[i]] << " "
          << weight[i] << endl;
    }
  }
}

void apply_edge_map(ifstream &in, ofstream &out, string edge_map_file) {
  vector<int> edge_map;
  ifstream in_edge_map(edge_map_file);
  int ind;
  while (in_edge_map >> ind) {
    edge_map.push_back(ind);
  }

  int u, v;
  while (in >> u >> v) {
    if (edge_map[u] == -1 || edge_map[v] == -1) {
      continue;
    }
    out << edge_map[u] << " " << edge_map[v] << endl;
  }
}

/*
 * Check if the input edgelist is symmetric
 */
bool is_symmetric(ifstream &in) {
  string first_line;
  getline(in, first_line);
  stringstream ss(first_line);
  int num_count_per_line = 0;
  string num;
  while (getline(ss, num, ' ')) {
    cout << num << endl;
    num_count_per_line++;
  }

  // reset cursor
  in.clear();
  in.seekg(0, ios::beg);

  set<edge_t> edges;
  set<edge_t> reversed_edges;
  int u, v;
  float w;
  if (num_count_per_line == 2) {
    cout << "Input file has 2 numbers per line, treat as un-weighted." << endl;
    while (in >> u >> v) {
      edges.insert(make_pair(u, v));
      reversed_edges.insert(make_pair(v, u));
    }
  } else if (num_count_per_line == 3) {
    cout << "Input file has 3 numbers per line, treat as weighted." << endl;
    while (in >> u >> v >> w) {
      edges.insert(make_pair(u, v));
      reversed_edges.insert(make_pair(v, u));
    }
  }
  return edges == reversed_edges;
}

/* Check if the inpu edgelist forms a single connected component*/
int is_connected(string in, bool print_comp0 = false) {
  int pid, status;
  if (pid = fork()) {
    waitpid(pid, &status, 0);
  } else {
    if (print_comp0) {
      execl("/data3/chenyh/sparsification/workload/gapbs/cc", "cc", "-f",
            in.c_str(), "-n", "1", "-v", "-a", "-c", NULL);
    } else {
      execl("/data3/chenyh/sparsification/workload/gapbs/cc", "cc", "-f",
            in.c_str(), "-n", "1", "-v", "-a", NULL);
    }
  }
  return status;
}

int main(int argc, char *argv[]) {
  CmdArgs args(argc, argv);
  cout << "---------- utils ----------" << endl;
  cout << "Input file: " << args.inFname() << endl;
  cout << "Output file: " << args.outFname() << endl;
  cout << "Mode: " << args.mode() << endl;

  ifstream fin(args.inFname());
  ofstream fout(args.outFname());
  if (args.mode() == "duw2uduw" || args.mode() == "1") {
    duw_to_uduw(fin, fout);
  } else if (args.mode() == "uduw2duw" || args.mode() == "2") {
    uduw_to_duw(fin, fout);
  } else if (args.mode() == "dw2udw" || args.mode() == "3") {
    dw_to_udw(fin, fout);
  } else if (args.mode() == "udw2dw" || args.mode() == "4") {
    udw_to_dw(fin, fout);
  } else if (args.mode() == "elim_disconnected_nodes" || args.mode() == "5") {
    elim_disconnected_nodes(fin, fout, args.outFname() + ".map");
  } else if (args.mode() == "sort_by_src_then_dst" || args.mode() == "6") {
    sort_by_src_then_dst(fin, fout);
  } else if (args.mode() == "zero_base_to_one_base" || args.mode() == "7") {
    zero_base_to_one_base(fin, fout);
  } else if (args.mode() == "is_symmetric" || args.mode() == "8") {
    cout << (is_symmetric(fin) ? "" : "Not ") << "Symmetric" << endl;
  } else if (args.mode() == "is_connected" || args.mode() == "9") {
    cout << is_connected(args.inFname()) << endl;
  } else if (args.mode() == "print_comp0" || args.mode() == "10") {
    cout << is_connected(args.inFname(), true) << endl;
  } else if (args.mode() == "symmetrize" || args.mode() == "11") {
    symmetrize(fin, fout);
  } else if (args.mode() == "unweight" || args.mode() == "12") {
    unweight(fin, fout);
  } else if (args.mode() == "apply_edge_map" || args.mode() == "13") {
    apply_edge_map(fin, fout, args.mapFname());
  } else {
    cout << "Unknown mode: " << args.mode() << endl << endl;
    cout << "Available modes: (use number or mode name) " << endl;
    cout << "\t\t1: duw2uduw" << endl;
    cout << "\t\t2: uduw2duw" << endl;
    cout << "\t\t3: dw2udw" << endl;
    cout << "\t\t4: udw2dw" << endl;
    cout << "\t\t5: elim_disconnected_nodes" << endl;
    cout << "\t\t6: sort_by_src_then_dst" << endl;
    cout << "\t\t7: zero_base_to_one_base" << endl;
    cout << "\t\t8: is_symmetric" << endl;
    cout << "\t\t9: is_connected" << endl;
    cout << "\t\t10: print_comp0" << endl;
    cout << "\t\t11: symmetrize" << endl;
    cout << "\t\t12: unweight" << endl;
    cout << "\t\t13: apply_edge_map" << endl;
  }

  return 0;
}
