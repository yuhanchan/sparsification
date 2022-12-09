CC = g++
# FLAGS = -O3 -s -fPIC -fopenmp
FLAGS = -g -fopenmp
# FLAGS = -O3 -fopenmp

all: main


test:
	$(CC) $(FLAGS) test.cpp -o bin/test

ACD: 
	$(CC) $(FLAGS) ACD.cpp -o bin/ACD

main: main.o Matrix.h Laplacian.o
	$(CC) $(FLAGS) main.o Laplacian.o -o bin/main

Laplacian.o: Laplacian.h Matrix.h
	$(CC) $(FLAGS) -c Laplacian.cpp -o Laplacian.o

main.o: main.cpp
	$(CC) $(FLAGS) -c main.cpp -o main.o


clean:
	rm -f main *.o
