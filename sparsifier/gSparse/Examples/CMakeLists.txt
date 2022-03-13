# C++ Skeleton Header-only library 
# Building Project Demo/Examples

# Requires CMake 3.5
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (gSparse-Examples)

# Add a demo executable and link it against the library
add_executable(Example-1 Example-1.cpp)
target_link_libraries(Example-1
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
)

# Add a demo executable and link it against the library
add_executable(Example-2 Example-2.cpp)
target_link_libraries(Example-2
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
)


# Add a demo executable and link it against the library
add_executable(Example-3 Example-3.cpp)
target_link_libraries(Example-3
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
)

# Add a demo executable and link it against the library
add_executable(Example-4 Example-4.cpp)
target_link_libraries(Example-4
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
)

# Add a demo executable and link it against the library
add_executable(Example-5 Example-5.cpp)
target_link_libraries(Example-5
    Eigen3::Eigen
    gSparse::gSparse  # Header-only library
)
# Add Spectra header-only
target_include_directories(Example-5 PRIVATE ${PROJECT_SOURCE_DIR}/spectra/include)
