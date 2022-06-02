CC = g++
# FLAGS = -O3 -s -fPIC -fopenmp
FLAGS = -g -fopenmp

all: main


test:
	$(CC) $(FLAGS) test.cpp -o bin/test

main: main.o Matrix.h Laplacian.o
	$(CC) $(FLAGS) main.o Laplacian.o -o bin/main

Laplacian.o: Laplacian.h Matrix.h
	$(CC) $(FLAGS) -c Laplacian.cpp -o Laplacian.o

main.o: main.cpp
	$(CC) $(FLAGS) -c main.cpp -o main.o


clean:
	rm -f main *.o
