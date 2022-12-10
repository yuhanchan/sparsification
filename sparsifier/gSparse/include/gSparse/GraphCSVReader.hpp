// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>,
// He Sun <he.sun@ed.ac.uk> This file is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution.

#ifndef GSPARSE_GRAPHCSVREADER_HPP
#define GSPARSE_GRAPHCSVREADER_HPP

#include <cstddef>   // size_t
#include <exception> // Runtime_exception
#include <fstream>   // File IO
#include <sstream>   // Errror Message
#include <string>    // Getline
#include <utility>   // std::move
#include <vector>    // load_csv

#include "Config.hpp"                // Library configuration
#include "Interface/GraphReader.hpp" // Baseclass definitions

namespace gSparse {
//! A Graph CSV Data Reader
/*!
    This class reads Graph Edge and List CSV files and transform them into Eigen
   Matrix. Node index starts from zero. Weight data type is gSparse::PRECISION
   type defined in Config.hpp
*/
class GraphCSVReader : public IGraphReader {
public:
  // Disallow default constructor
  GraphCSVReader() = delete;
  // Disallow copy constructor

  //! A copy Constructor
  GraphCSVReader(const GraphCSVReader &csvReader) noexcept {
    _delim = csvReader._delim;
    _edgeFile = csvReader._edgeFile;
    _weightFile = csvReader._weightFile;
  }
  //! A = operator overloaded
  GraphCSVReader &operator=(const GraphCSVReader &csvReader) noexcept {
    _delim = csvReader._delim;
    _edgeFile = csvReader._edgeFile;
    _weightFile = csvReader._weightFile;
    return *this;
  }
  //! Constructor
  /*!
  \param EdgeFileName: A filename pointing to a CSV file that contains Edge
  list. \param WeightFileName: A filename pointing to a CSV file that contains
  weight list. Default is "None": all weights equal to one. \param delimeter: A
  delimeter of the CSV file. Default is a comma (,).
  */
  GraphCSVReader(const std::string &EdgeFileName,
                 const std::string &WeightFileName = "None",
                 const std::string &delimeter = ",")
      : _delim(delimeter[0]), _edgeFile(EdgeFileName),
        _weightFile(WeightFileName) {}
  //! Read graph data from a CSV file specified in the constructor
  /*!
  \param Edges: An Eigen Matrix to receive the Edge list.
  \param Weights: An Eigen Matrix to receive the Weight list.
  */
  virtual void Read(gSparse::EdgeMatrix &Edges,
                    gSparse::PrecisionRowMatrix &Weights) {
    // Load data from CSV file and store into Edges
    load_csv<gSparse::EdgeMatrix>(_edgeFile, Edges);

    // Load data from CSV file and store into Edges. If weight file is "None",
    // set weights to one.
    if (_weightFile != "None")
      load_csv<gSparse::PrecisionRowMatrix>(_weightFile, Weights);
    else
      Weights = gSparse::PrecisionRowMatrix::Ones(Edges.rows(), 1);
  }

  //! Default destructor
  ~GraphCSVReader() = default;

private:
  char _delim;             //!< CSV file delimeter
  std::string _edgeFile;   //!< Edge file name
  std::string _weightFile; //!< Weight file name
  //! Template function that load CSV data into Eigen Matrix
  /*!
  \param path: path to filename.
  */
  template <typename M> void load_csv(const std::string &path, M &matrix) {
    // Allocate input file stream
    std::ifstream indata;
    indata.open(path);

    // Unable to open the file, throw exceptions
    if (!indata.is_open()) {
      std::stringstream ss;
      ss << "GraphCSVReader: File Not Found: " << path << std::endl;
      throw std::runtime_error(ss.str());
    }

    std::string line;                       // variable to hold each line
    std::vector<typename M::Scalar> values; // final value
    std::size_t rows = 0;                   // row counter

    // Read each line and add the data into vector . O(rc) complexity. r = rows.
    // c = columns.
    while (std::getline(indata, line)) {
      std::stringstream lineStream(line);
      std::string cell;
      while (std::getline(lineStream, cell, _delim)) {
        values.push_back(static_cast<typename M::Scalar>(std::stod(cell)));
      }
      ++rows;
    }
    // Close file
    indata.close();
    // Map STL vector to Eigen Matrix
    matrix =
        Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                                       M::ColsAtCompileTime, Eigen::RowMajor>>(
            values.data(), rows, values.size() / rows);
  }
};
} // namespace gSparse
#endif
