#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROOT_PROCESS 0

void read_matrix(double **matrix, int *n, int rank) {
    /**
     * reads matrix from user input
    */

    // read I/O for array size if is root process
    if (rank == 0) {
        scanf("%d", n);
    }
    MPI_Bcast(n, 1, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);

    // allocate augmented matrix
    matrix = (double **) malloc((2 * (*n)) * sizeof(double *));
    if (matrix == NULL) {
        fprintf(stderr, "Matrix row allocation failed\n");
        return;
    }

    for (int i = 0; i < *n; i++) {
        matrix[i] = (double *) malloc(2 * (*n) * sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Matrix columns allocation failed\n");
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return;
        }
    }

    // read I/O if is root process
    if (rank == 0) {
        // input elements into matrix
        double d = 0.0;
        for (int i = 0; i < *n; i++) {
            for (int j = 0; j < *n; j++) {
                scanf("%lf", &d);
                matrix[i][j] = d;
            }
        }
        // identity matrix
        for(int i = 0; i < *n; ++i)
        {
            for(int j = 0; j < 2*(*n); ++j)
            {
                if(j == (i+(*n)))
                {
                    matrix[i][j] = 1;
                }else{
                    matrix[i][j] = 0;
                }
            }
        }
    }    
}

int parse_matrix(double **matrix, int rank, char *filename) {
    // // TODO: read matrix from file
    // if (rank != 0) {
    //     return;
    // }
    // FILE *file;
    // int n,rows,cols;
    // int i, j;
    // file = fopen(*filename, "r");
    // if (file == NULL) {
    //     printf("Error opening the file.\n");
    //     return 1;
    // }
    //  // Read the number of elements from the first line
    // fscanf(file, "%d", &n);

    // matrix = (double **) malloc((2*n) * sizeof(double *));
    // if (matrix == NULL) {
    //     fprintf(stderr, "Matrix row allocation failed\n");
    //     return;
    // }

    // // Read the matrix elements
    // rows = 0;
    // while (fscanf(file, "%d", &matrix[rows][0]) == 1) {
    //     for (j = 1; j < n; j++) {
    //         fscanf(file, "%d", &matrix[rows][j]);
    //     }
    //     rows++;
    // }

    // // Close the file
    // fclose(file);

    // //identity matrix
    // for(i = 0; i < n; ++i)
    // {
    //     for(j = 0; j < 2*n; ++j)
    //     {
    //         if(j == (i+n))
    //         {
    //             matrix[i][j] = 1;
    //         }else{
    //             matrix[i][j] = 0;
    //         }
    //     }
    // }
}

void print_result(double **mat, int n, int rank) {
    if (rank != 0) {
        return;
    }
    for (int i = n; i < 2*n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
}

void invert_matrix(double **mat, int n, int my_rank, int comm_sz) {
    int block_size = n / (comm_sz);
    int local_start_row = my_rank * block_size;
    int local_end_row = (my_rank == comm_sz - 1) ? n : (my_rank + 1) * block_size;    
    double *pivot_row;

    pivot_row = (double *) malloc(2 * n * sizeof(double));

    /**
     * Reduce to diagonal matrix
     * 
     * processes subtraction to local rows by pivot row
    */
    for (int i = 0; i < n; i++) {
        int bcast_sender_rank = (i / comm_sz < comm_sz - 1) ? i / comm_sz : comm_sz - 1;

        if (bcast_sender_rank == my_rank) {
            // populate pivot row with row i
            for (int l = 0; l < 2 * n; l++) {
                pivot_row[l] = mat[i][l];
            }            
        }

        // send/receive broadcast of pivot row
        MPI_Bcast(pivot_row, 2*n, MPI_DOUBLE, bcast_sender_rank, MPI_COMM_WORLD);
        
        /**
         * Subtraction of rows by pivot received from bcast
         * 
         * only subtract rows owned by process, between local_start_row and local_end_row
        */
        for (int j = local_start_row; j < local_end_row; j++) {
            // Subtract row that is not pivot
            if (j != i) {
                double d = mat[j][i] / pivot_row[i];  // assign d to ratio of mat[j][i]/ pivot[i][i]

                for (int k = 0; k < 2 * n; k++) {
                    mat[j][k] -= d * mat[i][k];
                    // subtract my assigned row with d * pivot row
                }
            }
        }
    }

    /**
     * Reduce to identity matrix
     * 
     * Divide each local row with pivot diagonal value to produce local trapezoid of solution
    */
    for (int i = local_start_row; i < local_end_row; i++) {
        // Assign d with diagonal
        int d = mat[i][i];

        for (int j = 0; j < 2 * n; j++) {
            // Divide each element in local rows
            mat[i][j] /= d;
        }
    }

    // allocate result matrix
    double **result_mat;

    result_mat = (double **) malloc((local_end_row-local_start_row) * sizeof(double *));
    if (result_mat == NULL) {
        fprintf(stderr, "result_mat row allocation failed\n");
        return;
    }

    for (int i = 0; i < local_end_row-local_start_row; i++) {
        result_mat[i] = (double *) malloc(n * sizeof(double));
        if (result_mat[i] == NULL) {
            fprintf(stderr, "result_mat columns allocation failed\n");
            for (int j = 0; j < i; j++) {
                free(result_mat[j]);
            }
            free(result_mat);
            return;
        }
    }

    for (int i = 0; i < (local_end_row-local_start_row); i++)
    {
        for (int j = n; j < 2*n; j++)
        {
            result_mat[i][j-n] = mat[local_start_row+i][j];
        }
    }
    


    /**
     * ROOT PROCESS
     * 
     * combine results of matrix inverse
    */
    double **gathered_mat;

    if (my_rank == 0) {
        gathered_mat = (double **) malloc(n * sizeof(double *));
        if (gathered_mat == NULL) {
            fprintf(stderr, "gathered_mat row allocation failed\n");
            return;
        }

        for (int i = 0; i < local_end_row-local_start_row; i++) {
            gathered_mat[i] = (double *) malloc(n * sizeof(double));
            if (gathered_mat[i] == NULL) {
                fprintf(stderr, "gathered_mat columns allocation failed\n");
                for (int j = 0; j < i; j++) {
                    free(gathered_mat[j]);
                }
                free(gathered_mat);
                return;
            }
        }
    }

    /**
        * Reduce to diagonal matrix
        * receives and broadcasts current pivot row
    */
    MPI_Gather(result_mat, (local_end_row-local_start_row)*2*n, MPI_DOUBLE, gathered_mat, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_rank == 0){
        print_result(gathered_mat, n, my_rank);
    }
}

int main(int argc, char* argv[]) {
    int i = 0, j = 0, k = 0, n = 0;
    int my_rank, comm_sz;
    double **mat;
    double d = 0.0;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /**
     * Read matrix from stdin or from file
     * root process broadcasts result matrix to child processes
     * 
     * readMatrix only reads if my_rank is root
    */
    read_matrix(mat, &n, my_rank);
    
    // TODO: read argc and argv for textfile for input matrix?

    // broadcast initial matrix data
    MPI_Bcast(&n, 1, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(mat, n * 2*n, MPI_DOUBLE, ROOT_PROCESS, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // time calculate
    invert_matrix(mat, n, my_rank, comm_sz);

    print_result(mat, n, my_rank);
    MPI_Finalize();

    return 0;
}

