#include "CmdArgs.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

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

void elim_disconnected_nodes(ifstream &in, ofstream &out) {
  cout << "Warn: elim_disconnected_nodes assumes row indices in sorted" << endl;
  vector<int> row_ind;
  vector<int> col_ind;

  int u, v;
  while (in >> u >> v) {
    row_ind.push_back(u);
    col_ind.push_back(v);
  }

  // make_heap(row_ind.begin(), row_ind.end());
  // make_heap(col_ind.begin(), col_ind.end());
  // sort_heap(row_ind.begin(), row_ind.end());
  // sort_heap(col_ind.begin(), col_ind.end());
  //
  vector<int> ind_map(row_ind[row_ind.size() - 1] + 1,
                      0); // this is to map the row indices to the new indices
  for (int i = 0; i < row_ind.size(); i++) {
    ind_map[row_ind[i]] = 1;
  }
  for (int i = 1; i < ind_map.size(); i++) {
    ind_map[i] += ind_map[i - 1];
  }
  for (int i = 0; i < ind_map.size(); i++) {
    ind_map[i] -= 1;
  }

  for (int i = 1; i < ind_map.size(); i++) {
    if (ind_map[i] == ind_map[i - 1]) {
      ind_map[i] = -1;
    }
  }
  ofstream out_file("elim_ind_map.txt");
  cout << ind_map.size() << endl;
  for (int i = 0; i < ind_map.size(); i++) {
    out_file << ind_map[i] << endl;
  }

  // output
  for (int i = 0; i < row_ind.size(); i++) {
    out << ind_map[row_ind[i]] << " " << ind_map[col_ind[i]] << endl;
  }
}

int main(int argc, char *argv[]) {
  CmdArgs args(argc, argv);
  cout << "---------- utils ----------" << endl;
  cout << "Input file: " << args.inFname() << endl;
  cout << "Output file: " << args.outFname() << endl;
  cout << "Mode: " << args.mode() << endl;

  ifstream fin(args.inFname());
  ofstream fout(args.outFname());
  if (args.mode() == "duw2uduw") {
    duw_to_uduw(fin, fout);
  } else if (args.mode() == "uduw2duw") {
    uduw_to_duw(fin, fout);
  } else if (args.mode() == "dw2udw") {
    dw_to_udw(fin, fout);
  } else if (args.mode() == "udw2dw") {
    udw_to_dw(fin, fout);
  } else if (args.mode() == "elim_disconnected_nodes") {
    elim_disconnected_nodes(fin, fout);
  } else {
    cout << "Unknown mode: " << args.mode() << endl;
  }

  return 0;
}
