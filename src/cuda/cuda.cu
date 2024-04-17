#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#define ROW_OP_THREAD 256

double *allocate_matrix(int n, bool isAugmented)
{
  int col = isAugmented ? 2 * n : n;
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
    for (int j = rows; j < rows * 2; j++)
    {
      printf("%lf ", mat[get_matrix_index(i, j, rows * 2)]);
    }
    printf("\n");
  }
}

__device__ int GetMatrixIdx(int row, int col, int width)
{
  return width * row + col;
}

__global__ void NormalizePivotRow(double *mat, int n, int pivot_idx, double d)
{
  // called for pivot row only
  __shared__ double lmat[ROW_OP_THREAD];
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  if (col < 2 * n)
  {
    lmat[threadIdx.x] = mat[GetMatrixIdx(pivot_idx, col, 2 * n)] / d;
    __syncthreads();
    mat[GetMatrixIdx(pivot_idx, col, 2 * n)] = lmat[threadIdx.x];
  }
}

__global__ void NormalizePivotHelper(double *mat, int n, int pivot_idx)
{
  double d = mat[GetMatrixIdx(pivot_idx, pivot_idx, 2 * n)];
  NormalizePivotRow<<<ceil(2.0 * n / double(ROW_OP_THREAD)), ROW_OP_THREAD>>>(mat, n, pivot_idx, d);
  __syncthreads();
}

__global__ void SubtractNonPivot(double *mat, int n, int row_idx, int pivot_idx, double d)
{
  // called for nonpivot only

  __shared__ double lmat[ROW_OP_THREAD];
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  if (col < 2 * n)
  {
    lmat[threadIdx.x] = mat[GetMatrixIdx(row_idx, col, 2 * n)] - (d * mat[GetMatrixIdx(pivot_idx, col, 2 * n)]);
    __syncthreads();
    mat[GetMatrixIdx(row_idx, col, 2 * n)] = lmat[threadIdx.x];
  }
}

__global__ void SubsNonPivotKernel(double *mat, int n, int pivot_idx, int block_size)
{
  int row_size = n / block_size;
  int start_row = (threadIdx.x * row_size);
  int end_row = start_row + row_size;

  for (int row = start_row; row < end_row; row++)
  {
    if (row != pivot_idx)
    {
      double d = mat[GetMatrixIdx(row, pivot_idx, 2 * n)] / mat[GetMatrixIdx(pivot_idx, pivot_idx, 2 * n)];
      SubtractNonPivot<<<ceil(2.0 * n / double(ROW_OP_THREAD)), ROW_OP_THREAD>>>(mat, n, row, pivot_idx, d);
    }
  }
}

void invert_matrix(int n, double *mat)
{
  double *d_mat;
  size_t size = n * n * 2 * sizeof(double);
  cudaMalloc((void **)&d_mat, size);
  cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);

  int block_size = n >= 1024 ? 1024 : n;
  dim3 dimBlock(block_size);
  dim3 dimGrid(1, 1);

  for (int i = 0; i < n; i++)
  {
    NormalizePivotHelper<<<1, 1>>>(d_mat, n, i);
    cudaDeviceSynchronize();

    SubsNonPivotKernel<<<dimGrid, dimBlock>>>(d_mat, n, i, block_size);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(mat, d_mat, size, cudaMemcpyDeviceToHost);

  cudaFree(d_mat);
}

int main(void)
{
  int n;
  scanf("%d", &n);

  double *mat = allocate_matrix(n, true);
  read_matrix(mat, n);

  invert_matrix(n, mat);

  print_result(mat, n);

  return 0;
}