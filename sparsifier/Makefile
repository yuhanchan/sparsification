CXX_FLAGS += -std=c++11 -O3 -Wall -I../workload/gapbs/src -L../workload/gapbs/src
PAR_FLAG = -fopenmp

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

KERNELS = degree_prune
SUITE = $(KERNELS) 

.PHONY: all
all: $(SUITE)

% : src/%.cc
	$(CXX) $(CXX_FLAGS) $< -o bin/$@


.PHONY: clean
clean:
	rm -f bin/$(SUITE) 
