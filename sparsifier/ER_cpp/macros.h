#ifndef _MACROS_H_
#define _MACROS_H_

// #define DEBUG // will print out a lot of things
// #define PRINT_MATRIX // print out the matrices
// #define PSEUDO_RANDOM // read random numbers from pre-generated file
// #define PRINT_REFF // print out the final result of ER
#define USE_OPENMP // Warn: This should not be used together with PSEUDO_RANDOM,
// as multiple threads reading a file will defeat the purpose of
// using pseudo-random numbers to compare with python version.
#define READ_LA // read python generate CSC format of laplacian matrix,
                // because cpp implementation of laplacian matrix is very slow

#endif
