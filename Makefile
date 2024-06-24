
CC=g++
CC_FLAGS=-O3 -march=native -DDOUBLE 

all: laplacian-openmp

.PHONY: laplacian-openmp

laplacian-openmp: laplacian-openmp.cpp
	$(CC) laplacian-openmp.cpp -o laplacian-openmp-$(CC) $(CC_FLAGS) -fopenmp

laplacian: laplacian-openmp.cpp
	$(CC) laplacian-openmp.cpp -o laplacian-$(CC) $(CC_FLAGS)


.PHONY: clean

clean: 
	rm laplacian-openmp-gcc laplacian-openmp-clang
