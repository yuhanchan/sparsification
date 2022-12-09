#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#define PSUEDO_RANDOM

using namespace std;

// struct declarations
struct DenseMatrix;
struct SparseMatrixCOO;
struct SparseMatrixCSC;
struct LLp;
struct LDLinv_t;
struct ApproxCholPQ_t;
struct ApproxCholPQElem_t;
struct LLmatp_t;

struct DenseMatrix {
  int n, m;
  vector<vector<double>> mat;
  DenseMatrix(int n, int m) {
    this->n = n;
    this->m = m;
    mat.resize(n);
    for (int i = 0; i < n; i++) {
      mat[i].resize(m);
    }
  }

  void set(int i, int j, double val) { mat[i][j] = val; }

  double get(int i, int j) { return mat[i][j]; }

  void print() {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        cout << mat[i][j] << " ";
      }
      cout << endl;
    }
  }
};

struct SparseMatrixCOO {
  int n;
  int m;
  int nnz;
  vector<int> row;
  vector<int> col;
  vector<double> val;

  // convert to dense
  DenseMatrix toDense() {
    DenseMatrix dense(n, m);
    for (int i = 0; i < nnz; i++) {
      dense.set(this->row[i], this->col[i], this->val[i]);
    }
    return dense;
  }

  // sort col and permute row in the same way
  // this make converting to CSC easier
  void sort_col_then_row() {
    vector<int> perm(nnz);
    iota(perm.begin(), perm.end(), 0);
    sort(perm.begin(), perm.end(), [&](int i, int j) {
      if (col[i] == col[j]) {
        return row[i] < row[j];
      }
      return col[i] < col[j];
    });
    vector<int> new_row(nnz);
    vector<int> new_col(nnz);
    vector<double> new_val(nnz);
    for (int i = 0; i < nnz; i++) {
      new_row[i] = row[perm[i]];
      new_col[i] = col[perm[i]];
      new_val[i] = val[perm[i]];
    }
    row = new_row;
    col = new_col;
    val = new_val;
  }

  // convert to CSC
  SparseMatrixCSC toCSC() {
    sort_col_then_row();
    SparseMatrixCSC csc(n, m, nnz);
    csc.row_ind = row;
    csc.val = val;
    csc.col_ptr.resize(m + 1);
    for (int i = 0; i < nnz; i++) {
      csc.col_ptr[col[i] + 1]++;
    }
    for (int i = 0; i < m; i++) {
      csc.col_ptr[i + 1] += csc.col_ptr[i];
    }
    return csc;
  }
};

struct SparseMatrixCSC {
  int n;               // number of rows
  int m;               // number of columns
  int nnz;             // number of non-zero elements
  vector<int> col_ptr; // col_ptr[i] is the index of the first element in the
                       // i-th column
  vector<int> row_ind; // row_ind[i] is the row index of the i-th element
  vector<double> val;  // val[i] is the value of the i-th element

  long size_in_bytes() {
    return sizeof(int) * (col_ptr.size() + row_ind.size()) +
           sizeof(double) * val.size();
  }

  SparseMatrixCSC Transpose() {
    SparseMatrixCSC B(this->m, this->n, this->nnz);
    B.col_ptr.resize(this->n + 2);
    // counter per row
    for (int i = 0; i < this->nnz; i++) {
      B.col_ptr[this->row_ind[i] + 2]++;
    }

    // build new col_ptr
    for (int i = 2; i < B.col_ptr.size(); i++) {
      B.col_ptr[i] += B.col_ptr[i - 1];
    }

    // build new row_ind and val
    for (int i = 0; i < this->m; ++i) {
      for (int j = this->col_ptr[i]; j < this->col_ptr[i + 1]; j++) {
        int new_index = B.col_ptr[this->row_ind[j] + 1]++;
        B.row_ind[new_index] = i;
        B.val[new_index] = this->val[j];
      }
    }
    B.col_ptr.pop_back();
    return B;
  }

  // convert to COO format
  SparseMatrixCOO to_coo() {
    SparseMatrixCOO coo;
    coo.n = n;
    coo.m = m;
    coo.nnz = nnz;
    coo.row.resize(nnz);
    coo.col.resize(nnz);
    coo.val.resize(nnz);
    for (int j = 0; j < m; j++) {
      for (int i = col_ptr[j]; i < col_ptr[j + 1]; i++) {
        coo.row[i] = this->row_ind[i];
        coo.col[i] = j;
        coo.val[i] = this->val[i];
      }
    }
    return coo;
  }

  // Constructor
  SparseMatrixCSC(int n, int m, int nnz) {
    this->n = n;
    this->m = m;
    this->nnz = nnz;
    col_ptr.resize(m + 1);
    row_ind.resize(nnz);
    val.resize(nnz);
  }

  SparseMatrixCSC(int n, int m, int nnz, vector<int> col_ptr,
                  vector<int> row_ind, vector<double> val) {
    this->n = n;
    this->m = m;
    this->nnz = nnz;
    this->col_ptr = col_ptr;
    this->row_ind = row_ind;
    this->val = val;
  }

  // SparseMatrixCSC(int n, int m, int nnz, vector<int>& row_ind, vector<int>&
  // col_ind, vector<double>& val) {
  //     assert(nnz == row_ind.size());
  //     assert(nnz == col_ind.size());
  //     assert(nnz == val.size());

  //     SparseMatrixCSC B(n, m, nnz);
  //     B.col_ptr.resize(m + 2);

  //     for (int i = 0; i < nnz; i++) {
  //         B.col_ptr[row_ind[i]+2]++;
  //     }

  //     for (int i = 2; i < B.col_ptr.size(); i++) {
  //         B.col_ptr[i] += B.col_ptr[i - 1];
  //     }

  //     for (int i = 0; i < nnz; i++) {
  //         int index = B.col_ptr[col_ind[i]+1]++;
  //         B.row_ind[index] = row_ind[i];
  //         B.val[index] = val[i];
  //     }

  //     B.col_ptr.pop_back();

  //     // return B;
  //     // this->n = n;
  //     // this->m = m;
  //     // this->nnz = nnz;

  //     // this->col_ptr.resize(m + 1, 0);
  //     // this->row_ind.reserve(nnz);
  //     // this->val.reserve(nnz);

  //     // for (int i = 0; i < nnz; i++) {

  //     // }
  //     return B;
  // }

  // Print the sparse SparseMatrixCSC in dense format
  void print_dense() {
    DenseMatrix dense = this->to_coo().toDense();
    dense.print();
  }

  // copy Constructor
  SparseMatrixCSC(const SparseMatrixCSC &other) {
    n = other.n;
    m = other.m;
    nnz = other.nnz;
    col_ptr = other.col_ptr;
    row_ind = other.row_ind;
    val = other.val;
  }

  // << operator overload
  friend ostream &operator<<(ostream &os, const SparseMatrixCSC &mat) {
    os << "n: " << mat.n << " m: " << mat.m << " nnz: " << mat.nnz << endl;
    os << "col_ptr: ";
    for (int i = 0; i < mat.col_ptr.size(); i++) {
      os << mat.col_ptr[i] << " ";
    }
    os << endl;
    os << "row_ind: ";
    for (int i = 0; i < mat.row_ind.size(); i++) {
      os << mat.row_ind[i] << " ";
    }
    os << endl;
    os << "val: ";
    for (int i = 0; i < mat.val.size(); i++) {
      os << mat.val[i] << " ";
    }
    os << endl;
    return os;
  }

  // a x v multiplication
  vector<double> mul(vector<double> &vec) {
    assert(vec.size() == this->m);
    vector<double> res(this->n, 0);
    for (int i = 0; i < this->m; i++) {
      for (int j = this->col_ptr[i]; j < this->col_ptr[i + 1]; j++) {
        res[this->row_ind[j]] += this->val[j] * vec[i];
      }
    }
    return res;
  }

  // wtedEdgeVertexMat
  SparseMatrixCSC wtedEdgeVertexMat() {
    vector<int> row_ind;
    vector<int> col_ind;
    vector<double> val;
    // cout << "reserving " << this->nnz << " for row_ind and col_ind" << endl;
    row_ind.reserve(this->nnz);
    col_ind.reserve(this->nnz);
    val.reserve(this->nnz);
    int nnz = this->nnz;

    int count = 0;
    for (int jj = 0; jj < this->m; jj++) {
      for (int ii = this->col_ptr[jj]; ii < this->col_ptr[jj + 1]; ii++) {
        int i = this->row_ind[ii];
        if (i < jj) {
          col_ind.push_back(count);
          col_ind.push_back(count++);
          row_ind.push_back(i);
          row_ind.push_back(jj);
          double v_ = sqrt(this->val[ii]);
          val.push_back(v_);
          val.push_back(-v_);
        }
      }
    }

    // cout << nnz << " " << row_ind.size() << " " << col_ind.size() << endl;
    // assert(nnz == row_ind.size());
    // assert(nnz == col_ind.size());
    // assert(nnz == val.size());

    SparseMatrixCSC B(n, count, nnz);
    B.col_ptr.resize(count + 2);

    for (int i = 0; i < nnz; i++) {
      B.col_ptr[col_ind[i] + 2]++;
    }

    for (int i = 2; i < B.col_ptr.size(); i++) {
      B.col_ptr[i] += B.col_ptr[i - 1];
    }

    for (int i = 0; i < nnz; i++) {
      int index = B.col_ptr[col_ind[i] + 1]++;
      B.row_ind[index] = row_ind[i];
      B.val[index] = val[i];
    }

    B.col_ptr.pop_back();
    return B;
  }

  void clear() { // clear to save memory
    n = 0;
    m = 0;
    nnz = 0;
    col_ptr.clear();
    row_ind.clear();
    val.clear();
  }
};

struct LLp {
  int row;
  float val;
  LLp *next;
  LLp *reverse;

  LLp() {
    this->row = 0;
    this->val = 0;
    next = this;
    reverse = this;
  }
  LLp(int row, float val) {
    this->row = row;
    this->val = val;
    next = this;
    reverse = this;
  }
  LLp(int row, float val, LLp *next) {
    this->row = row;
    this->val = val;
    this->next = next;
    reverse = this;
  }
  LLp(int row, float val, LLp *next, LLp *reverse) {
    this->row = row;
    this->val = val;
    this->next = next;
    this->reverse = reverse;
  }

  // << operator overloading
  friend ostream &operator<<(ostream &os, const LLp &p) {
    os << &p << ", (" << p.row << ", " << p.val << ")"
       << ", next -> " << p.next << ", reverse -> " << p.reverse;
    return os;
  }

  // for debugging
  void print_until_selfloop() {
    int count = 0;
    LLp *lastptr = this;
    while (lastptr->next != lastptr) {
      cout << lastptr->row << "(" << lastptr->reverse->row << ")->";
      lastptr = lastptr->next;
      if (++count > 50) {
        break;
      }
    }
    count += 1;
    cout << lastptr->row << "(" << lastptr->reverse->row << ")";
    if (lastptr->next != lastptr) {
      cout << "... (chain longer than 50)";
    } else {
      cout << " (" << count << ")";
    }
    cout << endl;
  }
};

struct LDLinv_t {
  vector<int> col;
  vector<int> colptr;
  vector<int> row_ind;
  vector<double> val;
  vector<double> d;

  // Constructor
  LDLinv_t(int n) {
    this->col.resize(n - 1, -1);
    this->colptr.resize(n, -1);
    this->d.resize(n, 0);
  }
};

struct LLmatp_t {
  int n;              // numbers of rows/nodes
  vector<int> degs;   // degrees of nodes
  vector<LLp *> cols; // cols of the matrix
  vector<LLp *> lles; // linked list elements

  // Constructor
  LLmatp_t(int n, vector<int> degs, vector<LLp *> cols, vector<LLp *> lles) {
    this->n = n;
    this->degs = degs;
    this->cols = cols;
    this->lles = lles;
  }

  // << operator overloading
  friend ostream &operator<<(ostream &os, const LLmatp_t &mat) {
    os << "n: " << mat.n << endl;
    os << "degs: ";
    for (int i = 0; i < mat.n; i++) {
      os << mat.degs[i] << " ";
    }
    os << endl;
    os << "cols: " << endl;
    for (int i = 0; i < mat.cols.size(); i++) {
      os << *mat.cols[i] << endl;
    }
    os << endl;
    os << "lles: " << endl;
    for (int i = 0; i < mat.lles.size(); i++) {
      os << *mat.lles[i] << endl;
    }
    os << endl;
    return os;
  }

  // get_ll_col
  int get_ll_col(int i, vector<LLp *> &colspace) {
    LLp *ll = this->cols[i];
    int len = 0;
    while (ll->next != ll) {
      if (ll->val > 0) {
        len++;
        if (len > colspace.size()) {
          colspace.push_back(ll);
        } else {
          colspace[len - 1] = ll;
        }
      }

      ll = ll->next;
    }

    if (ll->val > 0) {
      len++;
      if (len > colspace.size()) {
        colspace.push_back(ll);
      } else {
        colspace[len - 1] = ll;
      }
    }

    return len;
  }

  // compressCol
  int compressCol(vector<LLp *> &colspace, int len) {
    // sort colspace by row
    sort(colspace.begin(), colspace.begin() + len,
         [](LLp *a, LLp *b) { return a->row < b->row; });

    int ptr = -1;
    int currow = -1;

    for (int i = 0; i < len; i++) {
      if (colspace[i]->row != currow) {
        currow = colspace[i]->row;
        ptr++;
        colspace[ptr] = colspace[i];
      } else {
        colspace[ptr]->val += colspace[i]->val;
        colspace[i]->reverse->val = 0;
      }
    }

    sort(colspace.begin(), colspace.begin() + ptr + 1,
         [](LLp *a, LLp *b) { return a->val < b->val; });

    return ptr + 1;
  }

  void pop(){

  }

};

LLmatp_t LLmatp(SparseMatrixCSC &a) {
  int n = a.n;
  int m = a.nnz;
#ifdef DEBUG
  cout << "n: " << n << endl;
  cout << "m: " << m << endl;
#endif

  vector<int> degs = vector<int>(n, 0);

  SparseMatrixCSC a_copy = SparseMatrixCSC(a);
  for (int i = 0; i < a_copy.nnz; i++) {
    a_copy.val[i] = i;
  }
  vector<double> flips = a_copy.Transpose().val;

#ifdef DEBUG
  cout << "flips: ";
  for (int i = 0; i < flips.size(); i++) {
    cout << flips[i] << " ";
  }
  cout << endl;
#endif

  vector<LLp *> cols(n, nullptr);
  vector<LLp *> llelems(m, nullptr);

  for (int i = 0; i < n; i++) {
    degs[i] = a.col_ptr[i + 1] - a.col_ptr[i];

    int ind = a.col_ptr[i];
    int j = a.row_ind[ind];
    double v = a.val[ind];
    LLp *llpend = new LLp(j, v);
    LLp *next = llpend;
    llelems[ind] = llpend;
    for (int ind = a.col_ptr[i] + 1; ind < a.col_ptr[i + 1]; ind++) {
      j = a.row_ind[ind];
      v = a.val[ind];
      LLp *llp = new LLp(j, v, next);
      llelems[ind] = llp;
      next = llp;
    }
    cols[i] = next;
  }

  for (int i = 0; i < n; i++) {
    for (int ind = a.col_ptr[i]; ind < a.col_ptr[i + 1]; ind++) {
      llelems[ind]->reverse = llelems[flips[ind]];
    }
  }

  return LLmatp_t(n, degs, cols, llelems);
}

LDLinv_t approxchol(LLmatp_t &a) {
#ifdef PSUEDO_RANDOM
  ifstream fin("/data3/chenyh/sparsification/utils/uniform_random.txt");
#endif

  int n = a.n;

  LDLinv_t ldli = LDLinv_t(n);
  int ldli_row_ptr = 0;

  vector<double> d(n, 0);

  int it = 0;

  vector<LLp *> colspace(n, nullptr);
  vector<double> csumspace(n, 0);
  vector<double> vals(n, 0);

  while (it < n - 1) {

    int i = a.pop();

    ldli.col[it] = i;
    ldli.colptr[it] = ldli_row_ptr;

    it++;

    int len = a.get_ll_col(i, colspace);

    len = a.compressCol(colspace, len);

    double csum = 0;
    for (int ii = 0; ii < len; ii++) {
      vals[ii] = colspace[ii]->val;
      csum += colspace[ii]->val;
      csumspace[ii] = csum;
    }
    double wdeg = csum;

    double colScale = 1;

    for (int joffset = 0; joffset < len - 1; joffset++) {
      LLp *ll = colspace[joffset];
      double w = vals[joffset] * colScale;
      int j = ll->row;
      LLp *revj = ll->reverse;

      double f = w / wdeg;

      vals[joffset] = 0;

#ifdef PSUEDO_RANDOM
      double random_number;
      fin >> random_number;
      // if reaching end of file, reset to beginning
      if (fin.eof()) {
        fin.clear();
        fin.seekg(0, fin.beg);
      }
      double r =
          random_number * (csum - csumspace[joffset]) + csumspace[joffset];
#else
      double r =
          static_cast<double>(rand()) / RAND_MAX * (csum - csumspace[joffset]) +
          csumspace[joffset];
#endif

      int koff = lower_bound(csumspace.begin(), csumspace.begin() + len, r) -
                 csumspace.begin(); // csumspace is assumed to be sorted
      // int koff = searchsortedfirst(csumspace, r, len) - csumspace.begin(); //
      // csumspace is assumed to be sorted

      int k = colspace[koff]->row;

      double newEdgeVal = f * (1 - f) * wdeg;

      revj->row = k;
      revj->val = newEdgeVal;
      revj->reverse = ll;

      LLp *khead = a.cols[k];
      a.cols[k] = ll;
      ll->next = khead;
      ll->reverse = revj;
      ll->val = newEdgeVal;
      ll->row = j;

      colScale *= (1 - f);
      wdeg = wdeg * pow((1 - f), 2);

      ldli.row_ind.push_back(j);
      ldli.val.push_back(f);
      ldli_row_ptr++;
    }

    LLp *ll = colspace[len - 1];
    double w = vals[len - 1] * colScale;
    int j = ll->row;
    LLp *revj = ll->reverse;

    revj->val = 0;

    ldli.row_ind.push_back(j);
    ldli.val.push_back(1);
    ldli_row_ptr++;

    d[i] = w;
  }

  ldli.colptr[it] = ldli_row_ptr;

  ldli.d = d;

#ifdef PSUEDO_RANDOM
  fin.close();
#endif

  return ldli;
}

int main(){
    return 0;
}