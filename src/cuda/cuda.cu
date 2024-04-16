
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

double *allocate_matrix(int n, bool isAugmented)
{
  int col = isAugmented ? 2*n : n;
    double *mat = (double *)malloc(n * col * sizeof(double));

    if (mat == NULL)
    {
        printf("Memory allocation failed!");
        free(mat);
        return NULL;
    }

    return mat;
}

int get_matrix_index(int row, int col, int width)
{
    return width * row + col;
}

void read_matrix(double *matrix, int n)
{
    double d = 0.0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                scanf("%lf", &d);
                matrix[get_matrix_index(i, j, 2 * (n))] = d;
            }
        }
        for (int i = 0; i < n; ++i)
        {
            for (int j = n; j < 2 * (n); ++j)
            {
                if (j == (i + (n)))
                {
                    matrix[get_matrix_index(i, j, 2 * (n))] = 1;
                }
                else
                {
                    matrix[get_matrix_index(i, j, 2 * (n))] = 0;
                }
            }
        }
}

void print_result(double *mat, int rows)
{
    printf("%d\n", rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = rows; j < rows*2; j++)
        {
            printf("%lf ", mat[get_matrix_index(i, j, rows*2)]);
        }
        printf("\n");
    }
}

__device__ int GetMatrixIdx(int row, int col, int width)
{
    return width * row + col;
}

__global__ void SubsPivotKernel(double* mat, int n, int pivot_idx, int block_size) {
  int row_size = n /block_size;
  int start_row = (threadIdx.x * row_size);
  int end_row = start_row + row_size;

  for (int row = start_row; row < end_row; row++) {
    if (row == pivot_idx) {
      double pivot = mat[GetMatrixIdx(pivot_idx, pivot_idx, 2*n)];
      for (int col = 0; col < 2*n; col++) {
        mat[GetMatrixIdx(pivot_idx, col, 2*n)] /= pivot;
      }
    }

    __syncthreads();

    if (row != pivot_idx) {
      double d = mat[GetMatrixIdx(row, pivot_idx, 2*n)] / mat[GetMatrixIdx(pivot_idx, pivot_idx, 2*n)];
      for (int col = 0; col < 2*n; col++) {
        mat[GetMatrixIdx(row, col, 2*n)] -= (d * mat[GetMatrixIdx(pivot_idx, col, 2*n)]);
      }
    }

    __syncthreads();
  }
}

void invert_matrix(int n, double* mat) {
  double* d_mat;
  size_t size = n * n * 2 * sizeof(double);
  cudaMalloc((void**)&d_mat, size);
  cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);

  int block_size = n >= 1024 ? 1024 : n;
   dim3 dimBlock(block_size);
   dim3 dimGrid(1, 1);

   for (int i=0; i<n; i++) {
    SubsPivotKernel<<<dimGrid, dimBlock>>>(d_mat, n, i, block_size);
   }

   cudaMemcpy(mat, d_mat, size, cudaMemcpyDeviceToHost);

   cudaFree(d_mat);
}

int main(void) {
  int n;
  scanf("%d", &n);

  double* mat = allocate_matrix(n, true);
  read_matrix(mat, n);

  invert_matrix(n, mat);

  print_result(mat, n);

  return 0;
}
