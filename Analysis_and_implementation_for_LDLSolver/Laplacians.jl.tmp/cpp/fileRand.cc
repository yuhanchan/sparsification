#include "fileRand.h"
#include <vector>
#include <exception>
#include <fstream>

static std::vector<double> randstore;
static int cur = 0;
static int limit = 0;
void initFileRand(const std::string &file, int num) {
  std::ifstream is(file, std::ios::binary);

  randstore.resize(num);
  is.read((char *) randstore.data(), sizeof(double) * num);

  limit = num;
}

double filerand() {
  if (limit == cur) {
    throw std::runtime_error("rand() drained");
  }

  return randstore[cur++];
}