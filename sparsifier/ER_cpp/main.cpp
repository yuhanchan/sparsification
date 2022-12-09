#include "Laplacian.h"
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

using namespace std;

SparseMatrixCSC read_file(string filename, bool weights, bool header,
                          bool exchange_row_col);
SparseMatrixCSC read_la(string filename);
void compute_reff(vector<vector<double>> &Z, SparseMatrixCSC &mat);

SparseMatrixCSC read_file(string filename, bool weights = false,
                          bool header = false, bool exchange_row_col = false) {
  ifstream infile(filename);
  string line;
  int n = 0, m = 0, nnz = 0;

  if (header) {
    infile >> n >> m >> nnz;

    vector<int> row_ind(nnz);
    vector<int> col_ind(nnz);
    vector<double> val(nnz, 1);
    vector<int> col_ptr(m + 1);
    if (weights && exchange_row_col) {
      for (int i = 0; i < nnz; i++) {
        infile >> col_ind[i] >> row_ind[i] >> val[i];
      }
    } else if (weights && !exchange_row_col) {
      for (int i = 0; i < nnz; i++) {
        infile >> row_ind[i] >> col_ind[i] >> val[i];
      }
    } else if (!weights && exchange_row_col) {
      for (int i = 0; i < nnz; i++) {
        infile >> col_ind[i] >> row_ind[i];
      }
    } else if (!weights && !exchange_row_col) {
      for (int i = 0; i < nnz; i++) {
        infile >> row_ind[i] >> col_ind[i];
      }
    }

    for (int i = 0; i < nnz; i++) {
      col_ptr[col_ind[i] + 1]++;
    }
    for (int i = 1; i < m + 1; i++) {
      col_ptr[i] += col_ptr[i - 1];
    }
    infile.close();

    return SparseMatrixCSC(n, m, nnz, col_ptr, row_ind, val);
  } else {
    vector<int> row_ind;
    vector<int> col_ind;
    vector<double> val;
    vector<int> col_ptr;
    string line;

    if (weights && exchange_row_col) {
      while (getline(infile, line)) {
        stringstream ss(line);
        int col, row;
        double weight;
        ss >> col >> row >> weight;
        col_ind.push_back(col);
        row_ind.push_back(row);
        val.push_back(weight);
        n = max(n, row);
        m = max(m, col);
        nnz++;
      }
    } else if (weights && !exchange_row_col) {
      while (getline(infile, line)) {
        stringstream ss(line);
        int row, col;
        double weight;
        ss >> row >> col >> weight;
        col_ind.push_back(col);
        row_ind.push_back(row);
        val.push_back(weight);
        n = max(n, row);
        m = max(m, col);
        nnz++;
      }
    } else if (!weights && exchange_row_col) {
      int last_col = -1;
      int last_row = -1;
      while (getline(infile, line)) {
        stringstream ss(line);
        int col, row;
        ss >> col >> row;

        col_ind.push_back(col);
        row_ind.push_back(row);
        n = max(n, row);
        m = max(m, col);
        nnz++;
      }
      val.resize(nnz, 1);
    } else if (!weights && !exchange_row_col) {
      while (getline(infile, line)) {
        stringstream ss(line);
        int row, col;
        ss >> row >> col;
        col_ind.push_back(col);
        row_ind.push_back(row);
        n = max(n, row);
        m = max(m, col);
        nnz++;
      }
      val.resize(nnz, 1);
    }

    n++;
    m++; // fix 0 index
    col_ptr.resize(m + 1, 0);
    for (int i = 0; i < nnz; i++) {
      col_ptr[col_ind[i] + 1]++;
    }
    for (int i = 1; i < m + 1; i++) {
      col_ptr[i] += col_ptr[i - 1];
    }
    infile.close();

    return SparseMatrixCSC(n, m, nnz, col_ptr, row_ind, val);
  }
}

SparseMatrixCSC read_la(string filename) {
  string col_ptr_file = filename + ".lap.col_ptr";
  string row_ind_file = filename + ".lap.row_ind";
  string val_file = filename + ".lap.val";
  ifstream col_ptr_in(col_ptr_file);
  ifstream row_ind_in(row_ind_file);
  ifstream val_in(val_file);

  vector<int> col_ptr;
  vector<int> row_ind;
  vector<double> val;
  string line;

  while (getline(col_ptr_in, line)) {
    stringstream ss(line);
    int col_ptr_val;
    ss >> col_ptr_val;
    col_ptr.push_back(col_ptr_val);
  }
  while (getline(row_ind_in, line)) {
    stringstream ss(line);
    int row_ind_val;
    ss >> row_ind_val;
    row_ind.push_back(row_ind_val);
  }
  while (getline(val_in, line)) {
    stringstream ss(line);
    double val_val;
    ss >> val_val;
    val.push_back(val_val);
  }

  col_ptr_in.close();
  row_ind_in.close();
  val_in.close();

  int n = col_ptr.size() - 1;
  int m = n;
  int nnz = row_ind.size();

  return SparseMatrixCSC(n, m, nnz, col_ptr, row_ind, val);
}

void compute_reff(vector<vector<double>> &Z, SparseMatrixCSC &mat) {
  vector<vector<double>> res;
  res.resize(mat.nnz, vector<double>{0.0, 0.0, 0.0});
// cout << mat.nnz << " " << mat.col_ptr[mat.col_ptr.size()-1] << endl;
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(128)
#endif
  for (int i = 0; i < mat.m; i++) {
    vector<double> diff;
    diff.resize(Z.size());
    for (int j = mat.col_ptr[i]; j < mat.col_ptr[i + 1]; j++) {
      for (int k = 0; k < Z.size(); k++) {
        diff[k] = Z[k][mat.row_ind[j]] - Z[k][i];
      }
      vector<double> tmp{static_cast<double>(mat.row_ind[j]),
                         static_cast<double>(i), pow(norm(diff), 2)};
      res[j] = tmp;
    }
  }

#ifdef PRINT_REFF
  // sort res first by 1st column, then by 2nd column
  sort(res.begin(), res.end(),
       [](const vector<double> &a, const vector<double> &b) {
         return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
       });

  cout << "Reff: " << endl;
  for (int i = 0; i < res.size(); i++) {
    cout << res[i][0] << "->" << res[i][1] << ": " << res[i][2] << endl;
  }
#endif
}

int main(int argc, char *argv[]) {
  auto total_start = chrono::high_resolution_clock::now();
  auto start = chrono::high_resolution_clock::now();
  // SparseMatrixCSC mat = read_file(argv[1], true, true, false);
  SparseMatrixCSC mat = read_file(argv[1], /*weights=*/false, /*header=*/false,
                                  /*exchange_row_col=*/true);
  // SparseMatrixCSC mat = read_file(argv[1], false, false, false);
  auto end = chrono::high_resolution_clock::now();
  cout << "read edge list time: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " ms" << endl;
  cout << "input mat size: " << mat.size_in_bytes() << " Byte" << endl;

#ifdef READ_LA
  start = chrono::high_resolution_clock::now();
  SparseMatrixCSC la = read_la(argv[1]);
  end = chrono::high_resolution_clock::now();
  cout << "read laplacian time: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " ms" << endl;
  cout << "laplacian mat size: " << la.size_in_bytes() << " Byte" << endl;
#endif

  // // save mat.col_ptr to filename
  // string col_ptr_file =  "a.rowind.cpp";
  // ofstream col_ptr_out(col_ptr_file);
  // for (int i = 0; i < mat.row_ind.size(); i++) {
  //     col_ptr_out << mat.row_ind[i]+1 << endl;
  // }

  auto llmat = LLmatp(mat);
  // // save mat.col_ptr to filename
  // string col_ptr_file =  "a.degs.cpp";
  // ofstream col_ptr_out(col_ptr_file);
  // for (int i = 0; i < llmat.degs.size(); i++) {
  //     col_ptr_out << llmat.degs[i] << endl;
  // }

  // return 0;

  // // print first 10 of the SparseMatrixCSC col_ptr, row_ind, val
  // cout << "mat: " << endl;
  // for (int i = 0; i < 10; i++) {
  //     cout << mat.col_ptr[i] << " ";
  // }
  // cout << endl;
  // for (int i = 0; i < 10; i++) {
  //     cout << mat.row_ind[i] << " ";
  // }
  // cout << endl;
  // for (int i = 0; i < 10; i++) {
  //     cout << mat.val[i] << " ";
  // }

  // return 0;

#ifdef PRINT_MATRIX
  mat.print_dense();
#endif

  int n = mat.n;
  int k = round(4.0 * log(n));

// k = 1000;
#ifdef DEBUG
  cout << "k: " << k << endl << endl;
#endif

  start = chrono::high_resolution_clock::now();
  SparseMatrixCSC U = mat.wtedEdgeVertexMat(); // U is nxm
  end = chrono::high_resolution_clock::now();
  cout << "generate incidence matrix time: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " ms" << endl;
  cout << "incidence mat size: " << U.size_in_bytes() << " Byte" << endl;

#ifdef PRINT_MATRIX
  cout << "U: " << endl;
  U.print_dense();
  cout << endl;
#endif

  int m = U.m;

  // use rand to initialize R
  start = chrono::high_resolution_clock::now();
  vector<vector<double>> R(k, vector<double>(m, 0)); // R is kxm

#ifdef PSEUDO_RANDOM
  ifstream fin("/data3/chenyh/sparsification/utils/normal_random.txt");
#endif

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(16)
#else
  random_device rd{};
  mt19937 gen{rd()};
  normal_distribution<double> dist{0, 1};
#endif
  for (int i = 0; i < k; i++) {
#ifdef USE_OPENMP
    random_device rd{};
    mt19937 gen{rd()};
    normal_distribution<double> dist{0, 1};
#endif
    for (int j = 0; j < m; j++) {
#ifdef PSEUDO_RANDOM
      fin >> R[i][j];
      // if reaching end of file, rewind
      if (fin.eof()) {
        fin.clear();
        fin.seekg(0, fin.beg);
      }
#else
      R[i][j] = dist(gen); // make noraml distribution with mean 0 and std 1
#endif
    }
  }

#ifdef PSEUDO_RANDOM
  fin.close();
#endif

  end = chrono::high_resolution_clock::now();
  cout << "generate random projection matrix time: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " ms" << endl;
  cout << "random projection mat size: " << R.size() * R[0].size() * sizeof(int)
       << " Byte" << endl;

#ifdef PRINT_MATRIX
  // print R
  cout << "R: " << endl;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < m; j++) {
      cout << R[i][j] << " ";
    }
    cout << endl;
  }
  cout << endl;
#endif

  start = chrono::high_resolution_clock::now();
  vector<vector<double>> UR(k, vector<double>(n, 0));
  for (int i = 0; i < k; i++) {
    UR[i] = U.mul(R[i]);
  }
  end = chrono::high_resolution_clock::now();
  cout << "multiply U and R time: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " ms" << endl;

#ifdef PRINT_MATRIX
  // print UR
  cout << "UR: " << endl;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      cout << UR[i][j] << " ";
    }
    cout << endl;
  }
  cout << endl;
#endif

  // delete U and R to save memory
  U.clear();
  R.clear();

  start = chrono::high_resolution_clock::now();
#ifdef READ_LA
  vector<vector<double>> Z = approxchol_lapGreedy(mat, la, UR);
#else
  vector<vector<double>> Z = approxchol_lapGreedy(mat, UR);
#endif
  end = chrono::high_resolution_clock::now();
  cout << "approxchol_lapGreedy time: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " ms" << endl;

#ifdef PRINT_MATRIX
  cout << "Z: " << endl;
  for (int i = 0; i < Z.size(); i++) {
    for (int j = 0; j < Z[i].size(); j++) {
      cout << Z[i][j] << " ";
    }
    cout << endl;
  }
  cout << endl;
#endif

  start = chrono::high_resolution_clock::now();
  compute_reff(Z, mat);
  end = chrono::high_resolution_clock::now();
  cout << "compute_reff time: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " ms" << endl;

  auto total_end = chrono::high_resolution_clock::now();
  cout
      << endl
      << "total time: "
      << chrono::duration_cast<chrono::seconds>(total_end - total_start).count()
      << " s" << endl;
  return 0;
}
