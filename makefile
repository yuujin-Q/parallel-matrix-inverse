OUTPUT_FOLDER = bin

all: serial parallel

parallel:
	mpicc src/open-mpi/openmpi.c -g -Wall -o $(OUTPUT_FOLDER)/mpi

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial

scp:
	scp -i ~/k01-07 -r $(FILE_NAME) k01-07@4.145.183.206:~/

clean:
	find ./bin -type f -not -name .gitignore -delete