# This project requires at least CMake 3.5
cmake_minimum_required (VERSION 3.5 FATAL_ERROR)

# Set project name
project(gSparse-Lib)

# Create a header-only library (INTERFACE)
# Create alias lib::myLibrary so other projects can link to it 
add_library(${PROJECT_NAME} INTERFACE)
add_library(gSparse::gSparse ALIAS ${PROJECT_NAME})

# Specify where the header-only library is
target_include_directories( ${PROJECT_NAME}
    INTERFACE ${PROJECT_SOURCE_DIR}/
)

