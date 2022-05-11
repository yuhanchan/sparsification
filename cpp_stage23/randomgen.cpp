#include <iostream>
#include <string>
#include <random>
#include <fstream>

using namespace std;

int main(int argc, char const *argv[]){
    if (argc != 3){
        cout << "Usage: " << argv[0] << " <uniform/normal> <#num>" << endl;
        return 1;
    }

    string type = argv[1];
    int num = stoi(argv[2]);
    if (type == "uniform"){
        srand(0);
        ofstream outfile("uniform_random.txt");
        for (int i = 0; i < num; i++){
            outfile << static_cast<double>(rand()) / RAND_MAX << endl;
        }
        outfile.close();
    } else if (type == "normal"){
        ofstream outfile("normal_random.txt");
        default_random_engine generator(0);
        normal_distribution<double> distribution(0, 1);
        for (int i = 0; i < num; i++) {
            outfile << distribution(generator) << endl;
        }
        outfile.close();
    } else {
        cout << "Usage: " << argv[0] << " <uniform/normal> <#num>" << endl;
        return 1;
    }

    return 0;
}
