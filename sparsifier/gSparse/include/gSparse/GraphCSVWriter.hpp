// Copyright (C) 2018 Thanaphon Chavengsaksongkram <as12production@gmail.com>, He Sun <he.sun@ed.ac.uk>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef GSPARSE_GRAPHCSVWRITER_HPP
#define GSPARSE_GRAPHCSVWRITER_HPP

#include <exception>  // Runtime_exception
#include <fstream>    // File IO
#include <sstream>    // Errror Message
#include <vector>     // load_csv
#include <string>     // Getline
#include <cstddef>    // size_t
#include <utility>    // std::move

#include "Config.hpp" // Library configuration
#include "Interface/GraphWriter.hpp"  // Baseclass definitions

namespace gSparse
{
    //! A Graph CSV Data Writer 
    /*!
        This class writes Graph Edge and List CSV files based on given input 
        Node index starts from zero. Weight data type is gSparse::PRECISION type defined in Config.hpp
    */
	class GraphCSVWriter : public IGraphWriter
	{
	public:
        // Disallow default constructor
        GraphCSVWriter() = delete;

        //! A copy Constructor
        GraphCSVWriter(const GraphCSVWriter & csvReader) noexcept
        {
            _delim = csvReader._delim;
            _edgeFile = csvReader._edgeFile;
            _weightFile = csvReader._weightFile;
        }
        //! A = operator overloaded
        GraphCSVWriter& operator=(const GraphCSVWriter & csvReader) noexcept
        {
            _delim = csvReader._delim;
            _edgeFile = csvReader._edgeFile;
            _weightFile = csvReader._weightFile;
            return *this;
        }
        //! Constructor
        /*!
        \param EdgeFileName: A filename pointing to a CSV file that contains Edge list.
        \param WeightFileName: A filename pointing to a CSV file that contains weight list. Default is "None": all weights equal to one.
        \param delimeter: A delimeter of the CSV file. Default is a comma (,).
        */
		GraphCSVWriter(const std::string & EdgeFileName, 
        const std::string & WeightFileName = "None", 
        std::string delimeter = ","):
			_delim(delimeter),
			_edgeFile(EdgeFileName),
			_weightFile(WeightFileName)
		{}
        //! Write graph data to a CSV file specified in the constructor
        /*!
        \param graph: A graph object
        */
		virtual void inline Write(const gSparse::Graph & graph)
		{
            // Graph object must exist
			if (graph == nullptr)
			{
				throw std::invalid_argument("GraphCSVWriter: graph must not be NULL");
			} 
            // Illegal use of this function. Graph object should have weights and must be written. 
			if (_weightFile == "None")
            {
				throw std::invalid_argument("GraphCSVWriter: Unable to write weight without destination filename");
            }
            // Write the Edge List
			write_csv<gSparse::EdgeMatrix>(_edgeFile, graph->GetEdgeList());
            // Write the Weight List
			write_csv<gSparse::PrecisionRowMatrix>(_weightFile, graph->GetWeightList());
		}
        //! Write graph data to a CSV file specified in the constructor
        /*!
        \param Edges: An edge list to be written to a CSV file
        */
		virtual void inline Write(const gSparse::EdgeMatrix & Edges)
		{
			write_csv<gSparse::EdgeMatrix>(_edgeFile, Edges);
		}
        //! Write graph data to a CSV file specified in the constructor
        /*!
        \param Edges: An edge list to be written to a CSV file
        \param Weights: A weight list to be written to a CSV file
        */
		virtual void inline Write(const gSparse::EdgeMatrix & Edges,
			const gSparse::PrecisionRowMatrix & Weights)
		{
            // Error due unspecified weight file
			if (_weightFile == "None")
				throw std::invalid_argument("GraphCSVWriter: Unable to write weight without destination filename");
            // Write Edge list
			write_csv<gSparse::EdgeMatrix>(_edgeFile, Edges);
            // Write Weight list
			write_csv<gSparse::PrecisionRowMatrix>(_weightFile, Weights);
		}
		virtual ~GraphCSVWriter() = default;
	private:

		std::string _delim;       //!< CSV file delimeter
		std::string  _edgeFile;   //!< Edge file name
		std::string  _weightFile; //!< Weight file name

        //! Template function to write Eigen Matrix into CSV file
        /*!
        \param fileName: File name to write the data
        \param matrix: Eigen Matrix containing the data
        */
		template <typename M>
		void inline write_csv(const std::string & fileName, const M & matrix)
		{
			Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, _delim, "\n");
			std::ofstream file(fileName.c_str());
			if (!file.is_open())
			{
				std::stringstream ss;
				ss << "GraphCSVWriter: File Not Found: " << fileName << std::endl;
				throw std::runtime_error(ss.str());
			}
			file << matrix.format(CSVFormat);
            file << "\n";
			file.close();
		}
	};
}
#endif
