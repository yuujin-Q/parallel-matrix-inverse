// gcc mp.c --openmp -o mp

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

double **allocate_matrix(int n, bool isAugmented)
{
    double **mat = (double **)malloc(n * sizeof(double *));

    if (mat == NULL)
    {
        printf("Memory allocation failed!");
        return NULL;
    }

    for (int i = 0; i < n; i++)
    {
        int rows = isAugmented ? 2 * n : n;
        mat[i] = (double *)malloc(rows * sizeof(double));
        if (mat[i] == NULL)
        {
            printf("Memory allocation failed!");
            // Free already allocated memory to avoid memory leaks
            for (int j = 0; j < i; j++)
            {
                free(mat[j]);
            }
            free(mat);
            return NULL;
        }
    }

    return mat;
}

double *get_partial_pivot(double **matrix, int n, int pivot_row, int local_row)
{
    double *result = (double *)malloc(2 * n * sizeof(double));
    if (result == NULL)
    {
        printf("Memory allocation failed!");
        return NULL;
    }

    double d = matrix[local_row][pivot_row] / matrix[pivot_row][pivot_row];
    if (d == 0)
    {
        for (int k = 0; k < 2 * n; k++)
        {
            result[k] = matrix[local_row][k];
        }
        return result;
    }
    for (int k = 0; k < 2 * n; k++)
    {
        double elim = d * matrix[pivot_row][k];
        result[k] = matrix[local_row][k] - elim;
    }
    return result;
}

double *get_reduced_row(double **matrix, int n, int local_row, double diagonal)
{
    double *result = (double *)malloc(2 * n * sizeof(double));
    if (result == NULL)
    {
        printf("Memory allocation failed!");
        return NULL;
    }
    for (int k = 0; k < 2 * n; k++)
    {
        result[k] = matrix[local_row][k] / diagonal;
    }
    return result;
}

void read_matrix(double **matrix, int n)
{
    /**
     * reads matrix from user input
     */
    // input elements into matrix
    double d = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            scanf("%lf", &d);
            matrix[i][j] = d;
        }
    }
    // augmented identity matrix
    for (int i = 0; i < n; ++i)
    {
        for (int j = n; j < 2 * n; ++j)
        {
            if (j == (i + n))
            {
                matrix[i][j] = 1;
            }
            else
            {
                matrix[i][j] = 0;
            }
        }
    }
}

void print_result(double **mat, int n)
{
    printf("%d\n", n);
    for (int i = 0; i < n; i++)
    {
        for (int j = n; j < 2 * n; j++)
        {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int n = 0;
    double **mat = NULL;

    int thread_count = strtol(argv[1], NULL, 0);

    scanf("%d", &n);
    mat = allocate_matrix(n, true);
    if (mat == NULL)
    {
        return 1;
    }
    read_matrix(mat, n);

    double start_time = omp_get_wtime();

    for (int i = 0; i < n; i++)
    {
        #pragma omp parallel for num_threads(thread_count)
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                double *new_row = get_partial_pivot(mat, n, i, j);
                
                #pragma omp critical
                {
                    for (int k = 0; k < 2 * n; k++)
                    {
                        mat[j][k] = new_row[k];
                    }
                }
                free(new_row);
            }
        }
    }

    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < n; i++)
    {
        double diagonal = mat[i][i];
        double *new_row = get_reduced_row(mat, n, i, diagonal);
        
        #pragma omp critical
        {
            for (int k = 0; k < 2 * n; k++)
            {
                mat[i][k] = new_row[k];
            }
        }
    }
    double end_time = omp_get_wtime();

    print_result(mat, n);
    printf("Elapsed time: %lf seconds\n", end_time - start_time);
    free(mat);

    return 0;
}