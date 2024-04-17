OUTPUT_FOLDER = bin

all: serial parallel-mpi parallel-mp parallel-cuda

parallel-mpi:
	mpicc src/open-mpi/openmpi.c -g -Wall -o $(OUTPUT_FOLDER)/mpi

parallel-mp:
	gcc src/open-mp/openmp.c --openmp -o $(OUTPUT_FOLDER)/mp

parallel-cuda:
	nvcc src/cuda/cuda.cu -rdc=true -lcudadevrt -o $(OUTPUT_FOLDER)/cuda

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial

scp:
	scp -i ~/k01-07 -r $(FILE_NAME) k01-07@4.145.183.206:~/

clean:
	find ./bin -type f -not -name .gitignore -delete