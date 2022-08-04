CC = g++
# FLAGS = -O3 -s -fPIC -fopenmp
FLAGS = -O3 -std=c++11

all: graphgen randomgen utils


graphgen:
	$(CC) $(FLAGS) graphgen.cpp -o bin/graphgen

randomgen:
	$(CC) $(FLAGS) randomgen.cpp -o bin/randomgen

utils:
	$(CC) $(FLAGS) utils.cpp CmdArgs.cpp -o bin/utils

clean:
	rm -f graphgen randomgen utils

