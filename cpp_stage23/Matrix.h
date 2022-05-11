#ifndef _MATRIX_H_
#define _MATRIX_H_

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

using namespace std;

int keyMap(int x, int n);
double dot(vector<double>& a, vector<double>& b);
double norm(vector<double>& a);
// y = a * x + y
void axpy2(double a, vector<double>& x, vector<double>& y);
// p = z + beta * p
void bzbeta(double beta, vector<double>& p, vector<double>& z);

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

    void set(int i, int j, double val) {
        mat[i][j] = val;
    }

    double get(int i, int j) {
        return mat[i][j];
    }

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
};

struct SparseMatrixCSC {
    int n; // number of rows
    int m; // number of columns
    int nnz; // number of non-zero elements
    vector<int> col_ptr; // col_ptr[i] is the index of the first element in the i-th column
    vector<int> row_ind; // row_ind[i] is the row index of the i-th element
    vector<double> val; // val[i] is the value of the i-th element
    
    SparseMatrixCSC Transpose() {
        SparseMatrixCSC B(this->m, this->n, this->nnz);
        B.col_ptr.resize(this->n + 2);
        // counter per row
        for (int i = 0; i < this->nnz; i++) {
            B.col_ptr[this->row_ind[i]+2]++;
        }
        
        // build new col_ptr
        for (int i = 2; i < B.col_ptr.size(); i++) {
            B.col_ptr[i] += B.col_ptr[i - 1];
        }
        
        // build new row_ind and val    
        for (int i = 0; i < this->m; ++i){
            for (int j = this->col_ptr[i]; j < this->col_ptr[i+1]; j++) {
                int new_index = B.col_ptr[this->row_ind[j]+1]++;
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
            for (int i = col_ptr[j]; i < col_ptr[j+1]; i++) {
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

    SparseMatrixCSC(int n, int m, int nnz, vector<int> col_ptr, vector<int> row_ind, vector<double> val) {
        this->n = n;
        this->m = m;
        this->nnz = nnz;
        this->col_ptr = col_ptr;
        this->row_ind = row_ind;
        this->val = val;
    }

    // SparseMatrixCSC(int n, int m, int nnz, vector<int>& row_ind, vector<int>& col_ind, vector<double>& val) {
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
    friend ostream& operator<<(ostream &os, const SparseMatrixCSC &mat) {
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
    vector<double> mul(vector<double>& vec) {
        assert(vec.size() == this->m);
        vector<double> res(this->n, 0);
        for (int i = 0; i < this->m; i++) {
            for (int j = this->col_ptr[i]; j < this->col_ptr[i+1]; j++) {
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
        row_ind.reserve(this->nnz);
        col_ind.reserve(this->nnz);
        val.reserve(this->nnz);
        int nnz = this->nnz;

        int count = 0;
        for (int jj = 0; jj < this->m; jj++) {
            for (int ii = this->col_ptr[jj]; ii < this->col_ptr[jj+1]; ii++) {
                int i = this->row_ind[ii];
                if(i<jj){
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

        assert(nnz == row_ind.size());
        assert(nnz == col_ind.size());
        assert(nnz == val.size());

        SparseMatrixCSC B(n, count, nnz);
        B.col_ptr.resize(count + 2);

        for (int i = 0; i < nnz; i++) {
            B.col_ptr[col_ind[i]+2]++;
        }
        
        for (int i = 2; i < B.col_ptr.size(); i++) {
            B.col_ptr[i] += B.col_ptr[i - 1];
        }

        for (int i = 0; i < nnz; i++) {
            int index = B.col_ptr[col_ind[i]+1]++;
            B.row_ind[index] = row_ind[i];
            B.val[index] = val[i];
        }
 
        B.col_ptr.pop_back();
        return B;
    }
};

struct LLp {
    int row;
    float val;
    LLp* next;
    LLp* reverse;

    LLp(){this->row = 0; this->val = 0; next = this; reverse = this;}
    LLp(int row, float val) { this->row = row; this->val = val; next = this; reverse = this; }
    LLp(int row, float val, LLp* next) { this->row = row; this->val = val; this->next = next; reverse = this; }
    LLp(int row, float val, LLp* next, LLp* reverse) { this->row = row; this->val = val; this->next = next; this->reverse = reverse; }

    // << operator overloading
    friend ostream& operator<<(ostream& os, const LLp& p) {
        os << &p <<  ", (" << p.row << ", " << p.val << ")" << ", next -> " << p.next << ", reverse -> " << p.reverse;
        return os;
    }
    
    // for debugging
    void print_until_selfloop() {
        int count = 0;
        LLp* lastptr = this;
        while(lastptr->next != lastptr) {
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

struct LDLinv_t{
    vector<int> col;
    vector<int> colptr;
    vector<int> row_ind;
    vector<double> val;
    vector<double> d;
    
    // Constructor
    LDLinv_t(int n){
        this->col.resize(n-1, -1);
        this->colptr.resize(n, -1);
        this->d.resize(n, 0);
    }
};

struct ApproxCholPQElem_t {
    int prev;
    int next;
    int key;
    // Constructor
    ApproxCholPQElem_t(int prev, int next, int key) {
        this->prev = prev;
        this->next = next;
        this->key = key;
    }

    // << operator overloading
    friend ostream& operator<<(ostream& os, const ApproxCholPQElem_t& p) {
        os <<  "prev: " << p.prev+1 << ", next: " << p.next+1 << ", key: " << p.key;
        return os;
    }
};

struct ApproxCholPQ_t{
    vector<ApproxCholPQElem_t*> elems;
    vector<int> lists;
    int minlist;
    int nitems;
    int n;

    // Constructor
    ApproxCholPQ_t(int n){
        this->elems.resize(n, nullptr);
        this->lists.resize(2*n+1, -1);
        this->minlist = 0;
        this->nitems = n;
        this->n = n;
    }

    // pop
    int pop() {
        if (nitems == 0) {
            cout << "Error: pop() called on empty PQ" << endl;
            return -1;
        }
        while(this->lists[this->minlist] == -1) {
            this->minlist++;
        } 
        int i = this->lists[this->minlist];
        int next = this->elems[i]->next;

        this->lists[this->minlist] = next;
        if(next > -1){
            this->elems[next] = new ApproxCholPQElem_t(-1, this->elems[next]->next, this->elems[next]->key);
        }

        this->nitems--;

        return i;
    }
    
    // Move
    void Move(int i, int newkey, int oldlist, int newlist){
        #ifdef DEBUG
        cout<< "Moved called with i: " << i << ", newkey: " << newkey << ", oldlist: " << oldlist << ", newlist: " << newlist << endl;
        #endif
        int prev = this->elems[i]->prev;
        int next = this->elems[i]->next;

        if(next > -1){
            this->elems[next] = new ApproxCholPQElem_t(prev, this->elems[next]->next, this->elems[next]->key);
        }
        if(prev > -1){
            this->elems[prev] = new ApproxCholPQElem_t(this->elems[prev]->prev, next, this->elems[prev]->key);
        } else {
            this->lists[oldlist] = next;
        }

        int head = this->lists[newlist];
        if(head > -1) {
            this->elems[head] = new ApproxCholPQElem_t(i, this->elems[head]->next, this->elems[head]->key);
        }
        this->lists[newlist] = i;

        this->elems[i] = new ApproxCholPQElem_t(-1, head, newkey);
    }

    // Dec
    void Dec(int i) {
        #ifdef DEBUG
        cout << "Dec called with i: " << i << endl;
        #endif
        int oldlist = keyMap(this->elems[i]->key, this->n) - 1;
        int newlist = keyMap(this->elems[i]->key - 1, this->n) - 1;
        
        if (newlist != oldlist){
            this->Move(i, this->elems[i]->key-1, oldlist, newlist);

            if(newlist < this->minlist){
                this->minlist = newlist;
            }
        } else {
            this->elems[i] = new ApproxCholPQElem_t(this->elems[i]->prev, this->elems[i]->next, this->elems[i]->key-1);
        }
    }

    // Inc
    void Inc(int i) {
        #ifdef DEBUG
        cout << "Inc called with i: " << i << endl;
        #endif
        int oldlist = keyMap(this->elems[i]->key, this->n) - 1;
        int newlist = keyMap(this->elems[i]->key + 1, this->n) - 1;
        
        if (newlist != oldlist){
            this->Move(i, this->elems[i]->key+1, oldlist, newlist);
        } else {
            this->elems[i] = new ApproxCholPQElem_t(this->elems[i]->prev, this->elems[i]->next, this->elems[i]->key+1);
        }
    }

    // << operator overloading
    friend ostream& operator<<(ostream& os, const ApproxCholPQ_t& pq) {
        os << "minlist = " << pq.minlist+1 << endl;
        os << "nitems = " << pq.nitems << endl; 
        os << "n = " << pq.n << endl; 
        os << "lists = ";
        for(int i = 0; i < pq.lists.size(); i++){
            os << pq.lists[i] + 1 << ", ";
        }
        os << endl;
        os << "elems = " << endl;
        for(int i = 0; i < pq.elems.size(); i++){
            os << i << ": " << *pq.elems[i] << endl;
        }
        return os;
    }
};

struct LLmatp_t{
    int n;              // numbers of rows/nodes
    vector<int> degs;   // degrees of nodes
    vector<LLp*> cols;  // cols of the matrix
    vector<LLp*> lles;  // linked list elements
    
    // Constructor
    LLmatp_t(int n, vector<int> degs, vector<LLp*> cols, vector<LLp*> lles) {
        this->n = n;
        this->degs = degs;
        this->cols = cols;
        this->lles = lles;
    }
    
    // << operator overloading
    friend ostream& operator<<(ostream& os, const LLmatp_t& mat) {
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
    int get_ll_col(int i, vector<LLp*>& colspace) {
        LLp* ll = this->cols[i];
        int len = 0;
        while (ll->next != ll) {
            if(ll->val > 0){
                len++;
                if(len > colspace.size()){
                    colspace.push_back(ll);
                } else {
                    colspace[len-1] = ll;
                }
            }
            
            ll = ll->next;
        }

        if(ll->val > 0){
            len++;
            if(len > colspace.size()){
                colspace.push_back(ll);
            } else {
                colspace[len-1] = ll;
            }
        }

        return len;
    }

    // compressCol
    int compressCol(vector<LLp*>& colspace, int len, ApproxCholPQ_t& pq){
        #ifdef DEBUG
        cout << "compressCol called with len: " << len << endl;
        #endif
        // sort colspace by row
        sort(colspace.begin(), colspace.begin() + len, [](LLp* a, LLp* b) { return a->row < b->row; });

        int ptr = -1;
        int currow = -1;

        for (int i = 0; i < len; i++) {
            if(colspace[i]->row != currow){
                currow = colspace[i]->row;
                ptr++;
                colspace[ptr] = colspace[i];
            } else {
                colspace[ptr]->val += colspace[i]->val;
                colspace[i]->reverse->val = 0;
                
                pq.Dec(currow);
            }
        }
        
        sort(colspace.begin(), colspace.begin() + ptr + 1, [](LLp* a, LLp* b) { return a->val < b->val; });

        return ptr + 1;
    }
    
    // for debugging
    void print_cols_until_selfloop() {
        for (int i = 0; i < this->cols.size(); i++) {
            cout << "" << i << ": ";
            this->cols[i]->print_until_selfloop();
        }
    }
};

#endif
