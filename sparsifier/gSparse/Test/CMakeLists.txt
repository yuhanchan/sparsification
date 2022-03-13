# Requires at least CMake 3.5
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (gSparse-Test-Suite)

# Download and build Google Test framework from github
add_subdirectory(google-test)

# enable CTest testing
enable_testing()

#####################################
# Add CSV Reader Test 
#####################################
add_executable(test-GraphCSVReader Test-GraphCSVReader.cpp)
# Link the test executable
target_link_libraries(test-GraphCSVReader
    GTest::GTest 
    GTest::Main
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
    --coverage
)
target_compile_options(test-GraphCSVReader PRIVATE --coverage)
add_test(NAME Test-CSVReader COMMAND test-GraphCSVReader)

#####################################
# Add CSV Writer Test 
#####################################
add_executable(test-GraphCSVWriter Test-GraphCSVWriter.cpp)
# Link the test executable
target_link_libraries(test-GraphCSVWriter
    GTest::GTest 
    GTest::Main
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
    --coverage
)
target_compile_options(test-GraphCSVWriter PRIVATE --coverage)
add_test(NAME Test-CSVWriter COMMAND test-GraphCSVWriter)

#####################################
# Add Undirected Graph Test
#####################################
add_executable(test-UndirectedGraph Test-UndirectedGraph.cpp)
# Link the test executable
target_link_libraries(test-UndirectedGraph
    GTest::GTest 
    GTest::Main
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
    --coverage
)
target_compile_options(test-UndirectedGraph PRIVATE --coverage)
add_test(NAME Test-UndirectedGraph COMMAND test-UndirectedGraph)

#####################################
# Add Utility/JL Test
#####################################
add_executable(test-Util-JL Test-Util-JL.cpp)
# Link the test executable
target_link_libraries(test-Util-JL
    GTest::GTest 
    GTest::Main
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
    --coverage
)
target_compile_options(test-Util-JL PRIVATE --coverage)
add_test(NAME Test-Util-JL COMMAND test-Util-JL)

#####################################
# Add ApproximateER Policy Test
#####################################
add_executable(test-ApproximateER Test-ApproximateER.cpp)
# Link the test executable
target_link_libraries(test-ApproximateER
    GTest::GTest 
    GTest::Main
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
    --coverage
)
target_compile_options(test-ApproximateER PRIVATE --coverage)
add_test(NAME Test-ApproximateER COMMAND test-ApproximateER)

#####################################
# Add ExactER Policy Test
#####################################
add_executable(test-ExactER Test-ExactER.cpp)
# Link the test executable
target_link_libraries(test-ExactER
    GTest::GTest 
    GTest::Main
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
    --coverage
)
target_compile_options(test-ExactER PRIVATE --coverage)
add_test(NAME Test-ExactER COMMAND test-ExactER)


#####################################
# Add Spectral Sparsifier by ER 
#####################################


add_executable(test-SpectralSparsifier-ERSampling Test-SpectralSparsifier-ERSampling.cpp)
# Link the test executable
target_link_libraries(test-SpectralSparsifier-ERSampling
    GTest::GTest 
    GTest::Main
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
    --coverage
)
target_compile_options(test-SpectralSparsifier-ERSampling PRIVATE --coverage)
add_test(NAME Test-SpectralSparsifier-ERSampling COMMAND test-SpectralSparsifier-ERSampling)

#####################################
# Add Builder Complete Graph
#####################################
add_executable(test-Builder-CompleteGraph Test-Builder-CompleteGraph.cpp)
# Link the test executable
target_link_libraries(test-Builder-CompleteGraph
    GTest::GTest 
    GTest::Main
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
    --coverage
)
target_compile_options(test-Builder-CompleteGraph PRIVATE --coverage)
add_test(NAME Test-Builder-CompleteGraph COMMAND test-Builder-CompleteGraph)


# Transfer files

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test-edges.csv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test-weight.csv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)


