CC = g++
# FLAGS = -O3 -s -fPIC -fopenmp
FLAGS = -g -fopenmp

all: main graphgen randomgen

graphgen:
	$(CC) $(FLAGS) graphgen.cpp -o graphgen

randomgen:
	$(CC) $(FLAGS) randomgen.cpp -o randomgen

main: main.o Matrix.h Laplacian.o
	$(CC) $(FLAGS) main.o Laplacian.o -o main

Laplacian.o: Laplacian.h Matrix.h
	$(CC) $(FLAGS) -c Laplacian.cpp -o Laplacian.o

main.o: main.cpp
	$(CC) $(FLAGS) -c main.cpp -o main.o


clean:
	rm -f graphgen randomgen main *.o
