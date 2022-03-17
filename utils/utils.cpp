#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include "CmdArgs.h"

using namespace std;

void dw_to_udw(ifstream & in, ofstream & out)
{
    int u, v;
    float w;    
    while (in >> u >> v >> w)
    {
        out << u << " " << v << " " << w << endl;
        out << v << " " << u << " " << w << endl;
    }
}

void udw_to_dw(ifstream & in, ofstream & out)
{
    int u, v;
    float w;    
    while (in >> u >> v >> w)
    {
        if (u < v)
        {
            out << u << " " << v << " " << w << endl;
        }
    }
}

void duw_to_uduw(ifstream & in, ofstream & out)
{
    int u, v;
    while (in >> u >> v)
    {
        out << u << " " << v << endl;
        out << v << " " << u << endl;
    }
}

void uduw_to_duw(ifstream & in, ofstream & out)
{
    int u, v;
    while (in >> u >> v)
    {
        if (u < v)
        {
            out << u << " " << v << endl;
        }
    }
}


int main(int argc, char* argv[])
{
    CmdArgs args(argc, argv);
    cout << "---------- utils ----------" << endl;
    cout << "Input file: " << args.inFname() << endl;
    cout << "Output file: " << args.outFname() << endl;
    cout << "Mode: " << args.mode() << endl;
    
    ifstream fin(args.inFname());
    ofstream fout(args.outFname());
    if(args.mode() == "duw2uduw"){
        duw_to_uduw(fin, fout);
    }
    else if(args.mode() == "uduw2duw"){
        uduw_to_duw(fin, fout);
    }
    else if(args.mode() == "dw2udw"){
        dw_to_udw(fin, fout);
    }
    else if(args.mode() == "udw2dw"){
        udw_to_dw(fin, fout);
    }
    else{
        cout << "Unknown mode: " << args.mode() << endl;
    }
 
    return 0;
}