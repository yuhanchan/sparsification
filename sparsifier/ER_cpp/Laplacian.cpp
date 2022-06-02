#include "Laplacian.h"
#include "macros.h"

double double_eps = 1e-9;

int keyMap(int x, int n){
    return x <= n ? x : n + x / n;
}

double dot(vector<double>& a, vector<double>& b){
    assert(a.size() == b.size());
    double sum = 0;
    for(int i = 0; i < a.size(); i++){
        sum += a[i] * b[i];
    }
    return sum;
}

double norm(vector<double>& a){
    return sqrt(dot(a, a));
}

// y = a * x + y
void axpy2(double a, vector<double>& x, vector<double>& y){
    assert(x.size() == y.size());
    for(int i = 0; i < x.size(); i++){
        y[i] += a * x[i];
    }
}

// p = z + beta * p
void bzbeta(double beta, vector<double>& p, vector<double>& z){
    assert(z.size() == p.size());
    for(int i = 0; i < z.size(); i++){
        p[i] = z[i] + beta * p[i];
    }
}

LLmatp_t LLmatp(SparseMatrixCSC& a){
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

    vector<LLp*> cols(n, nullptr);
    vector<LLp*> llelems(m, nullptr);

    for (int i = 0; i < n; i++) {
        degs[i] = a.col_ptr[i+1] - a.col_ptr[i];
        
        int ind = a.col_ptr[i];
        int j = a.row_ind[ind];
        double v = a.val[ind];
        LLp* llpend = new LLp(j, v);
        LLp* next = llpend;
        llelems[ind] = llpend;
        for (int ind = a.col_ptr[i] + 1; ind < a.col_ptr[i+1]; ind++) {
            j = a.row_ind[ind];
            v = a.val[ind];
            LLp* llp = new LLp(j, v, next);
            llelems[ind] = llp;
            next = llp;
        }
        cols[i] = next;
    }

    for (int i = 0; i < n; i++){
        for (int ind = a.col_ptr[i]; ind < a.col_ptr[i+1]; ind++) {
            llelems[ind]->reverse = llelems[flips[ind]];
        }
    }

    return LLmatp_t(n, degs, cols, llelems);
}

ApproxCholPQ_t ApproxCholPQ(vector<int>& degs){
    int n = degs.size();
    ApproxCholPQ_t res = ApproxCholPQ_t(n);

    for (int i = 0; i < n; i++) {
        int key = degs[i];
        int head = res.lists[key-1];

        if(head >= 0){
            res.elems[i] = new ApproxCholPQElem_t(-1, head, key);
            res.elems[head] = new ApproxCholPQElem_t(i, res.elems[head]->next, res.elems[head]->key);
        } else {
            res.elems[i] = new ApproxCholPQElem_t(-1, -1, key);
        }

        res.lists[key-1] = i;
    }

    return res;
}

LDLinv_t approxchol(LLmatp_t& a){
    #ifdef PSUEDO_RANDOM
    ifstream fin("/data3/chenyh/sparsification/utils/uniform_random.txt");
    #endif

    int n = a.n;

    LDLinv_t ldli = LDLinv_t(n);
    int ldli_row_ptr = 0;
    
    vector<double> d(n, 0);

    #ifdef DEBUG
    for (int i = 0; i < a.degs.size(); i++) {
        cout << "degs[" << i << "]: " << a.degs[i] << endl;
    }
    #endif

    ApproxCholPQ_t pq = ApproxCholPQ(a.degs);

    // // write a.degs to file
    // ofstream fout("a.degs.cpp");
    // for (int i = 0; i < a.degs.size(); i++) {
    //     fout << a.degs[i] << endl;
    // }
    // fout.close();
    

    #ifdef DEBUG
    cout << "pq: " << pq << endl;
    #endif

    int it = 0;

    vector<LLp*> colspace(n, nullptr);
    vector<double> csumspace(n, 0);
    vector<double> vals(n, 0);

    while(it < n-1){
        #ifdef DEBUG
        a.print_cols_until_selfloop();
        cout << "------------------------------------------------------------" << endl;
        cout << "pq1: " << pq << endl;
        #endif

        int i = pq.pop();

        #ifdef DEBUG
        cout << "it: " << it << ", i: " << i << endl << flush;
        #endif

        #ifdef DEBUG
        cout << "pq2: " << pq << endl;
        #endif

        ldli.col[it] = i;
        ldli.colptr[it] = ldli_row_ptr;

        it++;

        // cout << "i: " << i << ", row: " << a.cols[i]->row << ", ";
        int len = a.get_ll_col(i, colspace);

        // cout << it << ", len: " << len;
        len = a.compressCol(colspace, len, pq);

        // #ifdef DEBUG
        // cout << ", len: " << len << endl;
        // #endif

        #ifdef DEBUG
        cout << "pq3: " << pq << endl;
        #endif

        double csum = 0;
        for (int ii = 0; ii < len; ii++) {
            vals[ii] = colspace[ii]->val;
            csum += colspace[ii]->val;
            csumspace[ii] = csum;
        }
        double wdeg = csum;

        double colScale = 1;

        for(int joffset=0; joffset<len-1; joffset++){
            LLp* ll = colspace[joffset];
            double w = vals[joffset] * colScale;
            int j = ll->row;
            LLp* revj = ll->reverse;

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
            double r = random_number * (csum - csumspace[joffset]) + csumspace[joffset];
            #else
            double r = static_cast<double>(rand()) / RAND_MAX * (csum - csumspace[joffset]) + csumspace[joffset];
            #endif


            int koff = lower_bound(csumspace.begin(), csumspace.begin()+len, r) - csumspace.begin(); // csumspace is assumed to be sorted
            // int koff = searchsortedfirst(csumspace, r, len) - csumspace.begin(); // csumspace is assumed to be sorted

            int k = colspace[koff]->row;

            pq.Inc(k);

            #ifdef DEBUG
            cout << "pq: " << pq << endl;
            #endif

            double newEdgeVal = f * (1 - f) * wdeg;

            revj->row = k;
            revj->val = newEdgeVal;
            revj->reverse = ll;

            #ifdef DEBUG
            // insert to head of a.cols[k]
            cout << "insert to head of a.cols[" << k << "]" << endl;
            #endif
            LLp* khead = a.cols[k];
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
        
        #ifdef DEBUG
        cout << "pq4: " << pq << endl;
        #endif

        LLp* ll = colspace[len-1];
        double w = vals[len-1] * colScale;
        int j = ll->row;
        LLp* revj = ll->reverse;

        if(it < n-1) {
            pq.Dec(j);
        }

        #ifdef DEBUG
        cout << "pq5: " << pq << endl;
        #endif

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

void forwardSubstitution(LDLinv_t& ldli, vector<double>& y){
    for(int ii=0; ii<ldli.col.size(); ii++){
        int i = ldli.col[ii];

        int j0 = ldli.colptr[ii];
        int j1 = ldli.colptr[ii+1] - 1;

        double yi = y[i];

        for(int jj=j0; jj<j1; jj++){
            int j = ldli.row_ind[jj];
            y[j] += ldli.val[jj] * yi;
            yi *= (1.0 - ldli.val[jj]);
        }
        int j = ldli.row_ind[j1];
        y[j] += yi;
        y[i] = yi;
    }
}

void backwardSubstitution(LDLinv_t& ldli, vector<double>& y){
    for(int ii=ldli.col.size()-1; ii>=0; ii--){
        int i = ldli.col[ii];

        int j0 = ldli.colptr[ii];
        int j1 = ldli.colptr[ii+1] - 1;

        int j = ldli.row_ind[j1];
        double yi = y[i];
        yi += y[j];

        for(int jj=j1-1; jj>=j0; jj--){
            int j = ldli.row_ind[jj];
            yi = yi * (1.0 - ldli.val[jj]) + y[j] * ldli.val[jj];
        }
        y[i] = yi;
    }
}

vector<double> LDLSolver(LDLinv_t& ldli, vector<double>& b){
    // make a copy of b
    vector<double> y = b;

    forwardSubstitution(ldli, y);

    for (int i = 0; i < ldli.d.size(); i++) {
        if (ldli.d[i] != 0) {
            y[i] /= ldli.d[i];
        }
    }

    backwardSubstitution(ldli, y);

    double mu = accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
    for (int i = 0; i < y.size(); i++) {
        y[i] -= mu;
    }

    return y;
}

SparseMatrixCSC lap(SparseMatrixCSC& a){
    assert(a.n == a.m && "Matrix must be square");
    SparseMatrixCSC res = SparseMatrixCSC(a);

    // // print first 10 of res.col_ptr and res.row_ind
    // cout << "res.col_ptr: ";
    // for(int i=0; i<10; i++){
    //     cout << res.col_ptr[i] << " ";
    // }
    // cout << endl;
    // cout << "res.row_ind: ";
    // for(int i=0; i<10; i++){
    //     cout << res.row_ind[i] << " ";
    // }
    // cout << endl;

    // make laplacian matrix
    // #pragma omp parallel for
    for (int i = 0; i < res.m; i++) {
        double s = 0;
        for (int j = res.col_ptr[i]; j < res.col_ptr[i+1]; j++) {
            assert(res.row_ind[j] != i && "Diagonal element must be zero");
            s += res.val[j];
            res.val[j] = -res.val[j];
        }
        for(int j = res.col_ptr[i]; j < res.col_ptr[i+1];){
            if(res.row_ind[j] > i){
                res.row_ind.insert(res.row_ind.begin()+j, i);
                res.val.insert(res.val.begin()+j, s);
                res.nnz++;
                break;
            } else if(j == res.col_ptr[i+1]-1){
                res.row_ind.insert(res.row_ind.begin()+j+1, i);
                res.val.insert(res.val.begin()+j+1, s);
                res.nnz++;
                break;
            }
            j++;
        }
        if(res.col_ptr[i] != res.col_ptr[i+1]){
            for(int ii = i+1; ii < res.n+1; ii++){
                res.col_ptr[ii]++;
            }
        }
    }
    return res;
}

vector<double> pcg(SparseMatrixCSC& mat, vector<double> b, LDLinv_t& ldli, bool verbose=false, double tol=0.01, int maxits=1000, int stag_test=5){
    double al = 0;

    int n = mat.n;
    
    double nb = norm(b);

    if (nb == 0) {
        return vector<double>(n, 0);
    }
    
    vector<double> x(n, 0);
    vector<double> bestx(n, 0);
    double bestnr = 1.0;

    vector<double> r = b;
    vector<double> z = LDLSolver(ldli, r);
    vector<double> p = z;
    
    double rho = dot(r, z);
    double oldrho = rho;
    double best_rho = rho;
    int stag_count = 0;

    int itcnt = 0;
    while(itcnt < maxits){
        // cout << "itcnt: " << itcnt << endl;
        itcnt++;
        vector<double> q = mat.mul(p);

        double pq = dot(p, q);
       
        // I don't check for inf here, as it's tricky
        if ( -double_eps < pq && pq < double_eps) {
            if(verbose){
                cout<< "PCG stopped because pq is too small" << endl;
            }
            break;
        }

        al = rho / pq;

        if(al*norm(p) < double_eps*norm(x)){
            if(verbose){
                cout << "PCG stopped due to stagnation" << endl;
            }
            break;
        }

        axpy2(al, p, x);
        axpy2(-al, q, r);

        double nr = norm(r)/nb;
        if(nr < bestnr){
            bestnr = nr;
            bestx = x;
            best_rho = rho;
            stag_count = 0;
        }

        if(nr < tol){
            if(verbose){
                cout << "PCG converged in " << itcnt << " iterations" << endl;
            }
            break;
        }

        z = LDLSolver(ldli, r);

        oldrho = rho;
        rho = dot(r, z);

        if(rho < best_rho*(1-1/stag_test)){
            best_rho = rho;
            stag_count = 0;
        } else {
            if (stag_test > 0) {
                if(best_rho > (1-1/stag_test)*rho){
                    stag_count++;
                    if(stag_count > stag_test){
                        if(verbose){
                            cout << "PCG stopped due to stagnation test" << endl;
                        }
                        break;
                    }
                }
            }
        }

        if(rho < double_eps){
            if(verbose){
                cout << "PCG stopped due to rho being too small" << endl;
            }
            break;
        }

        double beta = rho / oldrho;
        if (beta < double_eps) {
            if(verbose){
                cout << "PCG stopped due to beta being too small" << endl;
            }
            break;
        }

        bzbeta(beta, p, z);
    }
    return bestx;
}

#ifdef READ_LA
vector<vector<double>> approxchol_lapGreedy(SparseMatrixCSC& a, SparseMatrixCSC& la, vector<vector<double>>& bs){
#else
vector<vector<double>> approxchol_lapGreedy(SparseMatrixCSC& a, vector<vector<double>>& bs){
#endif
    auto start = std::chrono::high_resolution_clock::now();
    #ifndef READ_LA
    SparseMatrixCSC la = lap(a);
    #endif
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Build laplacian: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << endl << flush;
    
    // auto start = std::chrono::high_resolution_clock::now();
    start = std::chrono::high_resolution_clock::now();
    LLmatp_t llmat = LLmatp(a);
    // auto end = std::chrono::high_resolution_clock::now();
    end = std::chrono::high_resolution_clock::now();
    cout << "LLmatp: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << endl;

    start = std::chrono::high_resolution_clock::now();
    LDLinv_t ldli = approxchol(llmat);
    end = std::chrono::high_resolution_clock::now();
    cout << "LDL: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << endl;
    
    vector<vector<double>> res(bs.size());

    auto s1 = chrono::high_resolution_clock::now();
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < bs.size(); i++) {
        auto s2 = chrono::high_resolution_clock::now();
        vector<double> b = bs[i];

        double b_mean = accumulate(b.begin(), b.end(), 0.0) / static_cast<double>(b.size());
        for (int j = 0; j < b.size(); j++) {
            b[j] -= b_mean;
        }
        
        res[i] = pcg(la, b, ldli);

        auto e2 = chrono::high_resolution_clock::now();
        cout << "PCG " << i << "/" << bs.size() << " time: " << chrono::duration_cast<chrono::milliseconds>(e2- s2).count() << " ms" << endl;
    }
    auto e1 = chrono::high_resolution_clock::now();
    #ifdef USE_OPENMP
    #pragma omp critical
    cout << "PCG total time: " << chrono::duration_cast<chrono::milliseconds>(e1 - s1).count() << " ms" << endl;
    #endif
    
    return res;
}
