#include <iostream>
#include <string>

#include "fileRand.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "need a file" << std::endl;
    return -1;
  }

  std::string file(argv[1]);
  initFileRand(file, 5000000);

  for (int i = 0; i < 10; i++) {
    double r = filerand();
    std::cout << r << std::endl;
  }
  return 0;
}