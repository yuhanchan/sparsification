#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <utility>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cout << "Usage: " << argv[0] << " [ring/dumbbell] [#node] [outfilename]" << endl;
        return 1;
    }

    string type = argv[1];
    int num_node = atoi(argv[2]);
    string outfilename = argv[3];

    vector<pair<int, int>> edge_list;
    if (type == "ring") {
        for (int i = 0; i < num_node - 1; i++) {
            edge_list.push_back(make_pair(i, i+1));
            edge_list.push_back(make_pair(i+1, i));
        }
        edge_list.push_back(make_pair(num_node - 1, 0));
        edge_list.push_back(make_pair(0, num_node - 1));
    } else if (type == "dumbbell") {
        for (int i = 0; i < int(num_node/2); i++) {
            for (int j = 0; j < int(num_node/2); j++) {
                if (i < j) {
                    edge_list.push_back(make_pair(i, j));
                    edge_list.push_back(make_pair(j, i));
                }
            }
        }

        for (int i = int(num_node/2); i < num_node; i++) {
            for (int j = int(num_node/2); j < num_node; j++) {
                if (i < j) {
                    edge_list.push_back(make_pair(i, j));
                    edge_list.push_back(make_pair(j, i));
                }
            }
        }
        edge_list.push_back(make_pair(int(num_node/2) - 1, int(num_node/2)));
        edge_list.push_back(make_pair(int(num_node/2), int(num_node/2) - 1));
    } else {
        cout << "Usage: " << argv[0] << " [ring/dumbbell] [#node] [outfilename]" << endl;
        return 1;
    }
    
    // do heap sort on edge_list, compare the sencond of pair
    make_heap(edge_list.begin(), edge_list.end(),
              [](const pair<int, int> &a, const pair<int, int> &b) {
                  if (a.second == b.second) {
                      return a.first < b.first;
                  } else {
                      return a.second < b.second;
                  }
              }
              );

    // sort the heap
    sort_heap(edge_list.begin(), edge_list.end(),
              [](const pair<int, int> &a, const pair<int, int> &b) {
                  if (a.second == b.second) {
                      return a.first < b.first;
                  } else {
                      return a.second < b.second;
                  }
              }
              );

    ofstream outfile;
    outfile.open(outfilename);
    outfile << num_node << " " << num_node << " " << edge_list.size() << endl;
    for (auto &e : edge_list) {
        outfile << e.first << " " << e.second << " 1" << endl;
    }
    outfile.close();
    return 0;

}

