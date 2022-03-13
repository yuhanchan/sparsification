# Requires at least CMake 3.5
cmake_minimum_required (VERSION 3.5)

##############################
# Project name
##############################
project(Skeleton-Header)

##############################
# Setting build options
##############################
# RELEASE Build Type
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build is Release")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif ()

# Options can be turned on and off by providing cmake with a -DOPTIONNAME=ON (or OFF) argument.
# For example, to turn off benchmarks, build a shared library, and use sanitize flags in a DEBUG build:
#    cmake .. -DCMAKE_BUILD_TYPE=DEBUG -DBUILD_DEMO=OFF -DBUILD_BENCH=ON

option(BUILD_DOC "Build documentation" ON)
option(BUILD_DEMO "Build demo" ON)
option(BUILD_BENCH "Build benchmark" ON)
option(BUILD_TEST "Build test" ON)

##############################
# Set additional compiler option
##############################
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##############################
# Dependencies
##############################
find_package(Eigen3)
if(Eigen3)
  MESSAGE(STATUS "Eigen 3 Found ${EIGEN3_INCLUDE_DIRS}")
else(Eigen3)
  MESSAGE(STATUS "Eigen3 Not Found. Potential Error")
endif(Eigen3)

##############################
# Add sub directories
##############################

# Header-only library
add_subdirectory(include)

# Test cases
if (BUILD_TEST)
  add_subdirectory(Test)
endif ()

# Documentation
if (BUILD_DOC)
  add_subdirectory(Doc)
endif ()

# Benchmark 
if (BUILD_BENCH)
  add_subdirectory(Benchmark)
endif ()

# Demo 
if (BUILD_DEMO)
  add_subdirectory(Examples)
endif ()

###################
# TODO: Packaging / Install Script / CPack goes here
###################












