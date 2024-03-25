#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

#define ROOT_PROCESS 0

int get_matrix_index(int row, int col, int width)
{
    return width * row + col;
}

void print_result(double *mat, int n, int m, int rank)
{
    /**
     * n = rows, m = cols
     */
    if (rank != 0)
    {
        return;
    }

    printf("%d\n", n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%lf ", mat[get_matrix_index(i, j, m)]);
        }
        printf("\n");
    }
}

int allocate_matrix(double **matrix, int *n, int rank, bool isAugmented)
{
    /**
     * allocates matrix
     * -1 rank for allocate only (n known)
     * 0+ rank for MPI processes
     */

    // read I/O for array size if is root process
    if (rank == 0)
    {
        scanf("%d", n);
    }
    if (rank != -1)
    {
        MPI_Bcast(n, 1, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // allocate
    if (isAugmented)
    {
        *matrix = (double *)malloc((*n) * (2 * (*n)) * sizeof(double));
    }
    else
    {
        *matrix = (double *)malloc((*n) * (*n) * sizeof(double));
    }
    if (*matrix == NULL)
    {
        fprintf(stderr, "Flat matrix allocation failed\n");
        return 1;
    }

    return 0;
}

void read_matrix(double **matrix, int *n, int rank)
{
    /**
     * reads matrix from user input
     */

    // read I/O if is root process
    if (rank == 0)
    {
        // input elements into matrix
        double d = 0.0;
        for (int i = 0; i < *n; i++)
        {
            for (int j = 0; j < *n; j++)
            {
                scanf("%lf", &d);
                (*matrix)[get_matrix_index(i, j, 2 * (*n))] = d;
            }
        }
        // augmented identity matrix
        for (int i = 0; i < *n; ++i)
        {
            for (int j = *n; j < 2 * (*n); ++j)
            {
                if (j == (i + (*n)))
                {
                    (*matrix)[get_matrix_index(i, j, 2 * (*n))] = 1;
                }
                else
                {
                    (*matrix)[get_matrix_index(i, j, 2 * (*n))] = 0;
                }
            }
        }
    }
}

int invert_matrix(double **mat, int n, int my_rank, int comm_sz, double **inverse, double *start)
{
    // calculate row start and ends for local processing
    int block_size = n / (comm_sz);
    int local_start_row = my_rank * block_size;
    int local_end_row = (my_rank == comm_sz - 1) ? n : (my_rank + 1) * block_size;
    double *pivot_row;

    // calculate recvcount and displs for gatherv & scatterv
    int recvcounts[comm_sz];
    int offsets[comm_sz];

    int sendcounts[comm_sz];
    int send_offset[comm_sz];
    if (my_rank == 0)
    {
        for (int i = 0; i < comm_sz; i++)
        {
            int process_start_row = i * (n / comm_sz);
            int process_end_row = (i == comm_sz - 1) ? n : (i + 1) * (n / comm_sz);

            recvcounts[i] = (process_end_row - process_start_row) * n;
            sendcounts[i] = (process_end_row - process_start_row) * 2 * n;

            offsets[i] = process_start_row * n;
            send_offset[i] = process_start_row * 2 * n;
        }
    }


    // scatterv for mat
    double *local_matrix = (double *)malloc((local_end_row - local_start_row) * 2 * n * sizeof(double));
    MPI_Scatterv(*mat, sendcounts, send_offset, MPI_DOUBLE, local_matrix, (local_end_row - local_start_row) * 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Start time after scatter
    *start = MPI_Wtime();

    pivot_row = (double *)malloc(2 * n * sizeof(double));

    /**
     * Reduce to diagonal matrix
     *
     * processes subtraction to local rows by pivot row
     */
    for (int i = 0; i < n; i++)
    {
        int bcast_sender_rank = (i / block_size < comm_sz) ? i / block_size : comm_sz - 1;

        if (bcast_sender_rank == my_rank)
        {
            // populate pivot row with row i
            for (int l = 0; l < 2 * n; l++)
            {
                pivot_row[l] = local_matrix[get_matrix_index(i - local_start_row, l, 2 * n)];
            }
        }

        // send/receive broadcast of pivot row
        MPI_Bcast(pivot_row, 2 * n, MPI_DOUBLE, bcast_sender_rank, MPI_COMM_WORLD);

        /**
         * Subtraction of rows by pivot received from bcast
         *
         * only subtract rows owned by process, between local_start_row and local_end_row
         */
        for (int j = 0; j < (local_end_row - local_start_row); j++)
        {
            // Subtract row that is not pivot
            if (j != i - local_start_row)
            {
                double d = local_matrix[get_matrix_index(j, i, 2 * n)] / pivot_row[i];

                if (d == 0)
                {
                    continue;
                }
                for (int k = 0; k < 2 * n; k++)
                {
                    local_matrix[get_matrix_index(j, k, 2 * n)] -= d * pivot_row[k];
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
    for (int i = 0; i < (local_end_row - local_start_row); i++)
    {
        // Assign d with diagonal
        double d = local_matrix[get_matrix_index(i, i + local_start_row, 2 * n)];

        for (int j = 0; j < 2 * n; j++)
        {
            // Divide each element in local rows
            local_matrix[get_matrix_index(i, j, 2 * n)] /= d;
        }
    }

    // allocate local_result matrix (fragment of final solution)
    double *local_result = (double *)malloc((local_end_row - local_start_row) * n * sizeof(double));
    if (local_result == NULL)
    {
        free(pivot_row);
        return 1;
    }
    for (int i = 0; i < (local_end_row - local_start_row); i++)
    {
        for (int j = n; j < 2 * n; j++)
        {
            local_result[get_matrix_index(i, j - n, n)] = local_matrix[get_matrix_index(i, j, 2 * n)];
        }
    }

    /**
     * RECOMBINE SOLUTION
     *
     * combine results of matrix inverse from local_results of each process to buffer `inverse`
     */
    if (my_rank == 0)
    {
        if (allocate_matrix(inverse, &n, -1, false) == 1)
        {
            free(pivot_row);
            free(local_result);
            return 1;
        }
    }
    MPI_Gatherv(local_result, (local_end_row - local_start_row) * n, MPI_DOUBLE,
                *inverse, recvcounts, offsets, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    return 0;
}

int main(int argc, char *argv[])
{
    int n = 0;
    int my_rank, comm_sz;
    double start, finish;
    double *mat = NULL;
    double *inverse = NULL;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /**
     * Read matrix from stdin or from file
     * root process broadcasts result matrix to child processes
     *
     * readMatrix only reads if my_rank is root
     */
    if (allocate_matrix(&mat, &n, my_rank, true) == 1)
        return 1;
    read_matrix(&mat, &n, my_rank);

    invert_matrix(&mat, n, my_rank, comm_sz, &inverse, &start);
    finish = MPI_Wtime();

    print_result(inverse, n, n, my_rank);

    if (my_rank == 0)
    {
        printf("Execution Time: %lf\n", finish - start);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    free(mat);
    free(inverse);

    return 0;
}
