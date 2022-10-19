#include <string>
#include <cstdint>
#include <iostream>
#include <algorithm>

#include "loadCSC.hpp"
#include "approxChol.hpp"
#include "fileRand.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "need a graph file name, a random file name and an output file" << std::endl;
    exit(-1);
  } else if (argc < 3) {
    std::cout << "need a random file name and an output file" << std::endl;
    exit(-1);
  } else if (argc < 4) {
    std::cout << "need an output file" << std::endl;
    exit(-1);
  }
  // initialize the filerand()
  std::string randfname = argv[2];
  initFileRand(randfname, 5000000);

  std::string name = argv[1];
  auto [spmat, flips] = readCSC<int64_t, double>(name);

  // std::cout << "------------------ CSC --------------------" << std::endl;
  // std::cout << spmat.m << " " << spmat.n << std::endl;

  // for (auto &e : spmat.colptr) {
  //   std::cout << e << " ";
  // }
  // std::cout << std::endl;
  // for (auto &e : spmat.rowval) {
  //   std::cout << e << " ";
  // }
  // std::cout << std::endl;
  // for (auto &e : spmat.nzval) {
  //   std::cout << e << " ";
  // }
  // std::cout << std::endl;

  // for (auto &e : flips) {
  //   std::cout << e << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "------------------ LLmatp --------------------" << std::endl;
  LLmatp<int64_t, double> llmatp(spmat, flips);

  // for (int i = 0; i < llmatp.cols.size(); i++) {
  //   std::cout << "deg: " << llmatp.degs[i] << std::endl;
  //   auto head = llmatp.cols[i];
  //   while (head != nullptr) {
  //     std::cout << "v: " << head->val << ", at: " << head->row << ", ptr to "
  //     << head->reverse->val << std::endl;
  //     head = head->next;
  //   }
  // }

  // std::cout << "------------------ order --------------------" << std::endl;
  // for (int i = 0; i < llmatp.cols.size(); i++) {
  //   std::cout << i << " ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < llmatp.cols.size(); i++) {
  //   std::cout << llmatp.degs[i] << " ";
  // }
  // std::cout << std::endl;

  std::vector<std::pair<int64_t, int64_t>> order_pairs;
  for (int i = 0; i < llmatp.cols.size(); i++) {
    order_pairs.push_back({llmatp.degs[i], i});
  }

  std::sort(order_pairs.begin(), order_pairs.end());
  std::vector<int64_t> order;
  for (auto &p : order_pairs) {
    order.push_back(p.second);
  }

  // std::cout << "---------------------------------------------" << std::endl;
  // for (auto &o : order) {
  //   std::cout << o << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "---------------------------------------------" << std::endl;
  auto ldli = approxChol_ordered(llmatp, order);
  // std::cout << "col: ";
  // for (auto c : ldli.col) {
  //   std::cout << c << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "colptr: ";
  // for (auto cptr : ldli.colptr) {
  //   std::cout << cptr << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "d: ";
  // for (auto d: ldli.d) {
  //   std::cout << d << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "fval: ";
  // for (auto f: ldli.fval) {
  //   std::cout << f << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "rv: ";
  // for (auto rv: ldli.rowval) {
  //   std::cout << rv << " ";
  // }
  // std::cout << std::endl;
  std::string ofname = argv[3];
  ldli.store(ofname);

  // std::cout << "------------------ add nodes to first --------------------" << std::endl;

  // llmatp.cols[0] = new LLp<int64_t, double>(2, 4.0, llmatp.cols[0]);
  // llmatp.cols[0] = new LLp<int64_t, double>(3, 4.0, llmatp.cols[0]);
  // llmatp.cols[0] = new LLp<int64_t, double>(7, 0, llmatp.cols[0]);
  // auto head = llmatp.cols[0];
  // while (head != nullptr) {
  //   std::cout << "v: " << head->val << ", at: " << head->row << ", ptr to "
  //   << head->reverse->val << std::endl;
  //   head = head->next;
  // }
  // llmatp.compress(0);
  // std::cout << "---- after " << std::endl;
  // head = llmatp.cols[0];
  // while (head != nullptr) {
  //   std::cout << "v: " << head->val << ", at: " << head->row << ", ptr to "
  //   << head->reverse->val << std::endl;
  //   head = head->next;
  // }
  // llmatp.sortByValue(0);
  // std::cout << "---- after2 " << std::endl;
  // head = llmatp.cols[0];
  // while (head != nullptr) {
  //   std::cout << "v: " << head->val << ", at: " << head->row << ", ptr to "
  //   << head->reverse->val << std::endl;
  //   head = head->next;
  // }
  // std::cout << "---- degs " << std::endl;
  // for (auto d : llmatp.degs) {
  //   std::cout << d << " ";
  // }
  // std::cout << std::endl;




}