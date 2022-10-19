#ifndef __LOAD_CSC_HPP__
#define __LOAD_CSC_HPP__
#include <vector>
#include <string>
#include <fstream>
#include <exception>
#include <utility>
#include <tuple>


template<typename Ti, typename Tv>
struct SpMatCSC {
  int64_t m;
  int64_t n;
  std::vector<Ti> colptr;
  std::vector<Ti> rowval;
  std::vector<Tv> nzval;
};


template<typename Ti, typename Tv>
std::tuple<SpMatCSC<Ti, Tv>, std::vector<Ti>> readCSC(const std::string &name) {
  std::ifstream is(name, std::ios::binary);
  int64_t m, n;
  is.read((char *) &m, sizeof(int64_t));
  is.read((char *) &n, sizeof(int64_t));

  SpMatCSC<Ti, Tv> spmat;
  spmat.m = m;
  spmat.n = n;

  spmat.colptr.resize(n + 1);
  is.read((char *) spmat.colptr.data(), sizeof(Ti) * (n + 1));
  for (auto &i : spmat.colptr) i--; // convert to 0-indexed
  

  int numnz = spmat.colptr[n];
  spmat.rowval.resize(numnz);
  spmat.nzval.resize(numnz);

  is.read((char *) spmat.rowval.data(), sizeof(Ti) * numnz);
  for (auto &r: spmat.rowval) r--; // convert to 0-index
  is.read((char *) spmat.nzval.data(), sizeof(Tv) * numnz);

  std::vector<Ti> flips(numnz, 0);
  is.read((char *) flips.data(), sizeof(Ti) * numnz);
  for (auto &f: flips) f--;

  return std::make_tuple(spmat, flips);
}

#endif //__LOAD_CSC_HPP__