#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include "Laplacian.h"

// #define DEBUG
// #define PRINT_MATRIX
// #define PSEUDO_RANDOM

using namespace std;

SparseMatrixCSC read_file(string filename, bool weights, bool header, bool exchange_row_col);
void compute_reff(vector<vector<double>>& Z, SparseMatrixCSC& mat);

SparseMatrixCSC read_file(string filename, bool weights=false, bool header=false, bool exchange_row_col=false) {
    ifstream infile(filename);
    string line;
    int n=0, m=0, nnz=0;

    if (header) {
        infile >> n >> m >> nnz;

        vector<int> row_ind(nnz);
        vector<int> col_ind(nnz);
        vector<double> val(nnz, 1);
        vector<int> col_ptr(m+1);
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
            col_ptr[col_ind[i]+1]++;
        }
        for (int i = 1; i < m+1; i++) {
            col_ptr[i] += col_ptr[i-1];
        }
        infile.close();

        return SparseMatrixCSC(n, m, nnz, col_ptr, row_ind, val);
    } else {
        vector<int> row_ind;
        vector<int> col_ind;
        vector<double> val;
        vector<int> col_ptr;
        string line;

        if(weights && exchange_row_col) {
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
                // if (col == 752 && row == 752) {
                //     cout << "here" << endl;
                // }
                // if (col < last_col) {
                //     cout << "input not ordered" << endl;
                //     cout << "col: " << col << " last_col: " << last_col << endl;
                // } else if (col > last_col) {
                //     last_col = col;
                //     last_row = -1;
                // }
                // if (row < last_row) {
                //     cout << "input not ordered" << endl;
                //     cout << "row: " << row << " last_row: " << last_row << endl;
                // } else if (row > last_row) {
                //     last_row = row;
                // }

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

        n++; m++; // fix 0 index
        col_ptr.resize(m+1, 0);
        for (int i = 0; i < nnz; i++) {
            col_ptr[col_ind[i]+1]++;
        }
        for (int i = 1; i < m+1; i++) {
            col_ptr[i] += col_ptr[i-1];
        }
        infile.close();

        // // output row_ind to row_ind.txt
        // ofstream outfile("row_ind.txt");
        // for (int i = 0; i < nnz; i++) {
        //     outfile << col_ind[i] << " " << row_ind[i] << endl;
        // }
        // outfile.close();
        // cout << n << " " << m << " " << nnz << " " << col_ptr[col_ptr.size()-1] << endl;
        // cout << "row_ind[80255]: " << row_ind[80255] << endl;
        return SparseMatrixCSC(n, m, nnz, col_ptr, row_ind, val);
    }
}

void compute_reff(vector<vector<double>>& Z, SparseMatrixCSC& mat){
    vector<vector<double>> res; 
    for (int i = 0; i < mat.m; i++) {
        for (int j = mat.col_ptr[i]; j < mat.col_ptr[i+1]; j++) {
            vector<double> diff(Z[0].size());
            for (int k = 0; k < Z.size(); k++) {
                diff[k] = Z[k][mat.row_ind[j]] - Z[k][i];
            }
            vector<double> tmp{static_cast<double>(mat.row_ind[j]), static_cast<double>(i), pow(norm(diff), 2)};
            res.push_back(tmp);
            // cout << mat.row_ind[j] << "->" << i << ": " << pow(norm(diff), 2) << endl;
        }
    }

    //sort res first by 1st column, then by 2nd column
    sort(res.begin(), res.end(), [](const vector<double>& a, const vector<double>& b) {
        return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
    });

    // print res
    cout << "Reff: " << endl;
    for (int i = 0; i < res.size(); i++) {
        cout << res[i][0] << "->" << res[i][1] << ": " << res[i][2] << endl;
    }
}


int main (int argc, char* argv[]) {
    // 0, 2, 3, 0, 1, 0
    // 2, 0, 5, 0, 7, 8
    // 3, 5, 0, 2.1, 0, 0
    // 0, 0, 2.1, 0, 7.7, 9.2
    // 1, 7, 0, 7.7, 0, 0
    // 0, 8, 0, 9.2, 0, 0
    // SparseMatrixCSC small_mat = SparseMatrixCSC(6, 6, 18, {0, 3, 7, 10, 13, 16, 18}, {1, 2, 4, 0, 2, 4, 5, 0, 1, 3, 2, 4, 5, 0, 1, 3, 1, 3}, {2, 3, 1, 2, 5, 7, 8, 3, 5, 2.1, 2.1, 7.7, 9.2, 1, 7, 7.7, 8, 9.2} );

    auto start = chrono::high_resolution_clock::now();
    // SparseMatrixCSC mat = read_file(argv[1], true, true, false);
    // SparseMatrixCSC mat = read_file(argv[1], false, false, true);
    SparseMatrixCSC mat = read_file(argv[1], true, true, false);
    auto end = chrono::high_resolution_clock::now();
    cout << "read_file time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    #ifdef PRINT_MATRIX
    mat.print_dense();
    #endif

    int n = mat.n;
    int k = round(4.0 * log(n));
    // k = 1000;
    cout << "k: " << k << endl << endl;

    start = chrono::high_resolution_clock::now();
    SparseMatrixCSC U = mat.wtedEdgeVertexMat(); // U is nxm
    end = chrono::high_resolution_clock::now();
    cout << "generate incidence matrix time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    #ifdef PRINT_MATRIX
    cout<< "U: " << endl;
    U.print_dense();
    cout<<endl;
    #endif

    int m = U.m;
    // use rand to initialize R
    start = chrono::high_resolution_clock::now();
    vector<vector<double>> R(k, vector<double>(m, 0)); // R is kxm
    random_device rd{};
    mt19937 gen{rd()};
    normal_distribution<double> dist{0, 1};

    #ifdef PSEUDO_RANDOM
    ifstream fin("normal_random.txt");
    #endif
    // #pragma omp parallel for num_threads(16)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            #ifdef PSEUDO_RANDOM
            fin >> R[i][j];
            #else
            R[i][j] = dist(gen); // make noraml distribution with mean 0 and std 1
            #endif
        }
    }
    #ifdef PSEUDO_RANDOM
    fin.close();
    #endif
    end = chrono::high_resolution_clock::now();
    cout << "generate random projection matrix time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

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
    for(int i = 0; i < k; i++){
        UR[i] = U.mul(R[i]);
    }
    end = chrono::high_resolution_clock::now();
    cout << "multiply U and R time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

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

    start = chrono::high_resolution_clock::now();
    vector<vector<double>> Z = approxchol_lapGreedy(mat, UR);
    end = chrono::high_resolution_clock::now();
    cout << "approxchol_lapGreedy time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

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
    cout << "compute_reff time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    return 0;
}
