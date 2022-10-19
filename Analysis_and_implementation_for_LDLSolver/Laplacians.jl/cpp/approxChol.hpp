#ifndef __APPROXCHOL_HPP__
#define __APPROXCHOL_HPP__

#include "loadCSC.hpp"
#include "fileRand.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>

template <typename Ti, typename Tv>
struct LLp {
  Ti row;
  Tv val;
  LLp<Ti, Tv> *next, *reverse;
  LLp(Ti r, Tv v, LLp<Ti, Tv> *n) : row(r), val(v), next(n) {}
};

// the LLmatp can actually be reduced to an array of linked list
template <typename Ti, typename Tv>
struct LLmatp {
  std::vector<LLp<Ti, Tv> *> cols;
  std::vector<Ti> degs;
  LLp<Ti, Tv> *dump;
  LLmatp(SpMatCSC<Ti, Tv> &mat, const std::vector<Ti> &flips) : 
    degs(mat.n, 0), cols(mat.n, nullptr), dump(nullptr) {
    
    std::vector<LLp<Ti, Tv> *> es;
    std::vector<Ti> ridx;

    for (Ti i = 0; i < mat.n; i++) {
      degs[i] = mat.colptr[i + 1] - mat.colptr[i];
      
      for (Ti idx = mat.colptr[i]; idx < mat.colptr[i + 1]; idx++) {
        cols[i] = new LLp(mat.rowval[idx], mat.nzval[idx], cols[i]);
        es.push_back(cols[i]);
        ridx.push_back(flips[idx]);
      }
    }

    for (int i = 0; i < es.size(); i++) {
      es[i]->reverse = es[ridx[i]];
    }
  }

  ~LLmatp() {
    for (auto &p: cols) {
      while(p != nullptr) { // remove all nodes in the cols
        auto n = p->next;
        delete p;
        p = n;
      }
    }

    while(dump != nullptr) { // remove all nodes in dump
      auto n = dump->next;
      delete dump;
      dump = n;
    }
  }

  // compress the column n, return the length of the compressed list
  // duplicated nodes are merged, and unnecessary nodes are moved to
  // dump
  // nodes with val == 0 are moved to dump
  Ti compress(Ti n) {
    if (cols[n] == nullptr) return 0; // no elements in this column, return

    // remove the val == 0's head
    auto head = cols[n];
    while (head && head->val <= 0) {
      cols[n] = head->next;
      head->next = dump;
      dump = head;
      head = cols[n];
    }
    if (!head) return 0; // cannot find a node with val > 0

    // first do an insertion sort to merge the elements with in the same row
    LLp<Ti, Tv> *sort = head->next, *prev = head;

    // compress the remaining nodes
    Ti num = 1;
    while (sort != nullptr) {
      // if the node is 0, then delete it
      if (sort->val <= 0) { // correspond to the get_ll_col()'s if ll.val > zero(Tval)
        // remove the node
        prev->next = sort->next;
        sort->next = dump;
        dump = sort;
        sort = prev->next;
        continue;
      } 

      // if the node is not 0, then search for an element
      // that is larger or equal to its row.
      // the algorithm is based on insertion sort, given that
      // most of the time there are not many elements
      // note: sort->row always == sort->row
      // note: bad memory access pattern
      LLp<Ti, Tv> *p = cols[n], **pp = &cols[n];
      while (p->row < sort->row) {
        pp = &(p->next);
        p = p->next;
      }

      if (p != sort) {
        if (p->row == sort->row) { // we can merge
          p->val += sort->val; // merge the value
          sort->reverse->val = 0; // remove the pair

          // remove the duplicated node & move forward
          prev->next = sort->next;
          sort->next = dump; 
          dump = sort;
          sort = prev->next;
        } else { // reorder the node
          prev->next = sort->next;
          sort->next = p;
          *pp = sort;

          sort = prev->next; // step next

          num++; // one more element in the sorted list
        }
      } else { // all elements before it has a smaller row
        prev = sort; // step next
        sort = sort->next;

        num++; // one more element in the sorted list
      }
    }

    degs[n] = num; // update the degree
    return num;
  }

  void sortByValue(Ti n) {
    if (cols[n] == nullptr) return; // no elements in this column, return

    // insertion sort (stable sort) using val as the key
    // note: bad memory access pattern
    LLp<Ti, Tv> *sort = cols[n]->next, *prev = cols[n];
    while (sort != nullptr) {
      LLp<Ti, Tv> *p = cols[n], **pp = &cols[n];
      // first sort by value, then by row
      while (p->val < sort->val || (p->val == sort->val && p->row < sort->row)) {
        pp = &(p->next);
        p = p->next;
      }

      if (p != sort) {
        prev->next = sort->next;
        sort->next = p;
        *pp = sort;
        sort = prev->next;
      } else {
        prev = sort;
        sort = sort->next;
      }
    }

  }
};

template <typename Ti, typename Tv>
struct LDLinv {
  std::vector<Ti> col, colptr, rowval;
  std::vector<Tv> fval, d;
  LDLinv(Ti n) : col(n - 1, 0), colptr(n, 0), d(n, 0) {}

  void store(const std::string &fname) {
    std::ofstream wf(fname, std::ios::out | std::ios::binary);
    int64_t csz, cptrsz, rvsz, fvsz, dsz;
    csz = col.size();
    cptrsz = colptr.size();
    rvsz = rowval.size();
    fvsz = fval.size();
    dsz = d.size();

    wf.write((char *) &csz, sizeof(int64_t));
    wf.write((char *) &cptrsz, sizeof(int64_t));
    wf.write((char *) &rvsz, sizeof(int64_t));
    wf.write((char *) &fvsz, sizeof(int64_t));
    wf.write((char *) &dsz, sizeof(int64_t));

    wf.write((char *) col.data(), sizeof(Ti) * csz);
    wf.write((char *) colptr.data(), sizeof(Ti) * cptrsz);
    wf.write((char *) rowval.data(), sizeof(Ti) * rvsz);
    wf.write((char *) fval.data(), sizeof(Tv) * fvsz);
    wf.write((char *) d.data(), sizeof(Tv) * dsz);
  }
};

template <typename Ti, typename Tv>
LDLinv<Ti, Tv> approxChol_ordered(LLmatp<Ti, Tv> &llmat, std::vector<Ti> &order) {
  // fix the seed for now
  auto n = llmat.cols.size(); // the number of nodes;

  LDLinv<Ti, Tv> ldli(n);

  std::vector<LLp<Ti, Tv> *> colspace(n, nullptr); // list of ptr (to refer)
  std::vector<Tv> cumspace(n, 0); // cumulates of value
  std::vector<Tv> vals(n, 0);
  
  Ti ldli_row_ptr = 0;

  for (Ti it = 0; it < n - 1; it++) { // n - 1 iterations [@inbounds while it < n]
    auto i = order[it];

    ldli.col[it] = i;
    ldli.colptr[it] = ldli_row_ptr;

    llmat.compress(i);
    llmat.sortByValue(i);

    // equiv. to get_ll_col() (get colspace), get vals, and compute cumspace
    auto head = llmat.cols[i];
    Ti num = 0;
    Tv csum = 0;
    // std::cout << i << ": ";
    while (head != nullptr) {
      vals[num] = head->val;
      // std::cout << "[" << head->val << " " << head->row << "] ";
      csum += head->val;
      cumspace[num] = csum; // cumspace is inclusive
      colspace[num] = head;
      head = head->next;
      num++;
    }
    // std::cout << std::endl;

    // std::cout << "at: " << i << ", num: " << num << std::endl;

    auto wdeg = csum;

    Tv colScale = 1;
    for (Ti j = 0; j < num - 1; j++) {
      auto nodeDel = colspace[j]; // select an edge to delete
      auto w = vals[j]; // no need to time the colScale

      auto f = w / wdeg;
      auto newEdgeVal = f * (1 - f) * wdeg * colScale;

      // take a random sample
      auto r = filerand() * (csum - cumspace[j]) + cumspace[j]; // r here is always < cumspace[num - 1]
      auto koff = std::upper_bound(cumspace.begin(), cumspace.begin() + num, r) - cumspace.begin();
      auto nodeK = colspace[koff];


      // establish an edge between nodeDel->row and nodeK->row
      auto sampledNodeRow = nodeK->row; // k
      // change the reverse to an edge to nodeK->row
      // assume current is i, we are removing (j <-> k)
      // here j->i is changed to j->k
      // note: there is no need to reset the link between this
      // pair of nodes.
      // std::cout << "(" << nodeDel->row << ", " << nodeK->row << "): " << newEdgeVal << std::endl;
      nodeDel->reverse->row = nodeK->row; // need to reset the row to k
      nodeDel->reverse->val = newEdgeVal;

      // here i->j is changed to k->j. need to move the node to k
      // insertion point is nodeK's reverse
      llmat.cols[i] = nodeDel->next;
      nodeDel->next = nodeK->reverse->next; // append it to the nodeK
      nodeK->reverse->next = nodeDel; 
      nodeDel->val = newEdgeVal;

      // update colscale and wdeg
      colScale *= (1 - f);
      wdeg -= w; // equiv to *= (1-f)

      ldli.rowval.push_back(nodeDel->row);
      ldli.fval.push_back(f);
      ldli_row_ptr++;
    }

    // the num can be 0 as the remaining graph may be an unconnected graph
    // the if statement skips for that case
    if (num > 0) {
      // for the last node, set the node on the other end to 0
      auto nodeDel = colspace[num - 1];
      nodeDel->reverse->val = 0;
      ldli.rowval.push_back(nodeDel->row);
      ldli.fval.push_back(1);
      ldli_row_ptr++;
      ldli.d[i] = vals[num - 1] * colScale;
    } else { // ldli is set to 1 as a default
      ldli.d[i] = 1;
    }
  }

  ldli.colptr[n - 1] = ldli_row_ptr;

  return ldli;
}

#endif