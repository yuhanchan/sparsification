#include "CmdArgs.h"
#include <cassert>
#include <iostream>

using namespace std;

CmdArgs::CmdArgs(int argc, char **argv) {
  _success = true; // Unless proven otherwise

  if (argc <= 1) {
    this->dispHelp();
    _success = false;
    return;
  }

  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-i") {
      i++;
      if (i == argc) {
        cout << "Missing in file after -i argument." << endl;
        _success = false;
        continue;
      }
      _inFname = argv[i];
    } else if (arg == "-u") {
      i++;
      if (i == argc) {
        cout << "Missing ref file after -u argument." << endl;
        _success = false;
        continue;
      }
      _refFname = argv[i];
    } else if (arg == "-o") {
      i++;
      if (i == argc) {
        cout << "Missing out file after -o argument." << endl;
        _success = false;
        continue;
      }
      _outFname = argv[i];
    } else if (arg == "-p") {
      i++;
      if (i == argc) {
        cout << "Missing out file after -p argument." << endl;
        _success = false;
        continue;
      }
      _mapFname = argv[i];
    } else if (arg == "-m") {
      i++;
      if (i == argc) {
        cout << "Missing out file after -m argument." << endl;
        _success = false;
        continue;
      }
      _mode = argv[i];
    } else {
      cout << "Unknown command line argument: " << arg << endl;
      _success = false;
    }
  }
}

void CmdArgs::dispHelp() const {
  cout << endl;
  cout << "--- utils ---" << endl;
  cout << endl;
  cout << "Input arguments:" << endl;
  cout << "  -i [path]" << endl;
  cout << "       input filename" << endl;
  cout << "Reference arguments:" << endl;
  cout << "  -r [path]" << endl;
  cout << "       ref filename" << endl;
  cout << "Output arguments:" << endl;
  cout << "  -o [path]" << endl;
  cout << "       output filename" << endl;
  cout << "Map arguments:" << endl;
  cout << "  -p [path]" << endl;
  cout << "       output filename" << endl;
  cout << "Mode arguments:" << endl;
  cout << "  -m [mode]" << endl;
  cout << "       duw2uduw / uduw2duw / dw2udw / udw2dw" << endl;
  cout << endl;
}
