// gcc mp.c --openmp -o mp

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

double** allocate_matrix(int n, bool isAugmented) {
    double** mat = (double**) malloc(n * sizeof(double*));

    if (mat == NULL) {
        printf("Memory allocation failed!");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        int rows = isAugmented ? 2*n : n;
        mat[i] = (double*) malloc(rows * sizeof(double));
        if (mat[i] == NULL) {
            printf("Memory allocation failed!");
            // Free already allocated memory to avoid memory leaks
            for (int j = 0; j < i; j++) {
                free(mat[j]);
            }
            free(mat);
            return NULL;
        }
    }

    return mat;
}

void read_matrix(double **matrix, int n)
{
    /**
     * reads matrix from user input
     */
    // input elements into matrix
    double d = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &d);
            matrix[i][j] = d;
        }
    }
    // augmented identity matrix
    for (int i = 0; i < n; ++i) {
        for (int j = n; j < 2 * n; ++j) {
            if (j == (i + n)) {
                matrix[i][j] = 1;
            } else {
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
        for (int j = 0; j < 2*n; j++)
        {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int n = 0;
    double **mat = NULL;
    
    int thread_count = strtol(argv[1], NULL, 0);

    scanf("%d", &n);
    mat = allocate_matrix(n, true);
    if (mat == NULL) {
        return 1;
    }
    read_matrix(mat, n);

    double *pivot_row;
    pivot_row = (double *)malloc(2 * n * sizeof(double));
    
    for (int i=0; i<n; i++) {
        pivot_row = mat[i];
        # pragma omp parallel for num_threads(thread_count)
        for (int j = 0; j < n; j++)
        {
            if (j != i) {   
                double d = mat[j][i] / pivot_row[i];
                if (d == 0) {
                    continue;
                }
                for (int k = 0; k < 2 * n; k++) {
                    double elim = d * pivot_row[k]; 
                    
                    #pragma omp critical
                    {
                        mat[j][k] -= elim;
                    }
                }
            }
        }
        print_result(mat, n);
    }
    free(pivot_row);

    # pragma omp parallel for num_threads(thread_count)
    for (int i=0; i<n; i++) {
        double diagonal = mat[i][i];
        for(int j = 0; j < 2*n; ++j)
        {
            double newValue = mat[i][j] / diagonal;
            
            #pragma omp critical
            mat[i][j] = newValue;
        }
    }

    print_result(mat, n);
    free(mat);

    return 0;
}