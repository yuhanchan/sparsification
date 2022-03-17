#include <gSparse/gSparse.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

int main(int argc, char* argv[])
{   
    if(argc != 6)
    {
        std::cout<<"Usage: "<<argv[0]<<" <epsilon> <input el path> <output el path> <output weight path> <er save path>"<<std::endl;
        return -1;
    }
    // Create a CSV Reader and specify the file location
    gSparse::GraphReader csvReader = std::make_shared<gSparse::GraphCSVReader>(argv[2], "None", " ");
    // Create an Undirected Graph. gSparse::Graph object is a shared_ptr.
    gSparse::Graph graph = std::make_shared<gSparse::UndirectedGraph>(csvReader);

    // Display original edge count
    std::cout<<"Original Edge Count: " << graph->GetEdgeCount() << std::endl;

    // Creating Sparsifier Object
    gSparse::SpectralSparsifier::ERSampling sparsifier(graph);
    // Set Hyper-parameters
    // Approximate the Effective Weight Resistance (Faster)
    // C = 4 and Epsilon = 0.5
    sparsifier.SetERPolicy(gSparse::SpectralSparsifier::APPROXIMATE_ER);
    // sparsifier.SetERPolicy(gSparse::SpectralSparsifier::EXACT_ER);
    sparsifier.SetC(4.0);
    sparsifier.SetEpsilon(std::stof(argv[1]));
    // Compute Effective Weight Resistance
    
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream file(argv[5]);
    if (file.good()) {
        std::cout <<"ER already computed, loading..." << std::endl;
        sparsifier.LoadER(file);
        file.close();
    } else {
        std::ofstream outfile(argv[5]);
        std::cout <<"ER not computed, computing..." << std::endl;
        sparsifier.Compute();
        sparsifier.SaveER(outfile);
        outfile.close();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Compute time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl;
    // Perform Sparsification
    start = std::chrono::high_resolution_clock::now();
    auto sparseGraph1 = sparsifier.GetSparsifiedGraph();
    end = std::chrono::high_resolution_clock::now();
    float prune_rate = 1.0 - float(sparseGraph1->GetEdgeCount())/float(graph->GetEdgeCount());
    std::cout<<"Epsilon = "<<argv[1]<<", Sparised Edge Count (ApproxER): " << sparseGraph1->GetEdgeCount() << " prunerate = " << prune_rate << std::endl;
    std::cout<<"GetSparsifiedGraph time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl;

    gSparse::GraphWriter csvWriter = std::make_shared<gSparse::GraphCSVWriter>(argv[3], argv[4], " ");
    
    start = std::chrono::high_resolution_clock::now();
    // csvWriter->Write(sparseGraph1->GetEdgeList(), sparseGraph1->GetWeightList());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Write time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl;
    return EXIT_SUCCESS;
}