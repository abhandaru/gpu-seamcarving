/*
    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "seam_carving.h"

#define BLOCK_WIDTH 32

namespace cuda {

  /* This function computes a cell of the matrix multiplication result
     depending on the thread and block indices of the current thread. */
  __global__
  void matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2,
      float *sq_matrix_result, int sq_dimension) {

    // Extract thread and block index information
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int row = by * BLOCK_WIDTH + ty;
    int col = bx * BLOCK_WIDTH + tx;

    float sum = 0;

    // Iterate over blocks of the input matrices
    int b = 0;
    for (; b < sq_dimension / BLOCK_WIDTH; b++) {

        // Allocate shared memory with padding to avoid bank conflicts
        __shared__ float shared_sq_matrix_1[BLOCK_WIDTH * (BLOCK_WIDTH + 1)];
        __shared__ float shared_sq_matrix_2[BLOCK_WIDTH * (BLOCK_WIDTH + 1)];

        // Each thread populates a cell of shared memory matrix 1
        // We must check that the thread remains within the bounds
        if (row < sq_dimension && b * BLOCK_WIDTH + tx < sq_dimension) {
            shared_sq_matrix_1[ty * (BLOCK_WIDTH + 1) + tx] =
                sq_matrix_1[row * sq_dimension + b * BLOCK_WIDTH + tx];
        }

        // Each thread populates a cell of shared memory matrix 2
        // We must check that the thread remains within the bounds
        if (col < sq_dimension && b * BLOCK_WIDTH + ty < sq_dimension) {
            shared_sq_matrix_2[ty * (BLOCK_WIDTH + 1) + tx] =
                sq_matrix_2[(b * BLOCK_WIDTH + ty) * sq_dimension + col];
        }

        // Ensure all threads have loaded memory performing computations
        __syncthreads();

        for (int i = 0; i < BLOCK_WIDTH; i++) {
            sum += shared_sq_matrix_1[ty * (BLOCK_WIDTH + 1) + i] *
                shared_sq_matrix_2[i * (BLOCK_WIDTH + 1) + tx];
        }

        // Ensure that all threads have completed computation before
        // reloading memory
        __syncthreads();
    }

    // Complete the computation and store the result if we are in the bounds
    // of the result matrix
    if (row < sq_dimension && col < sq_dimension) {
        for (int j = b * BLOCK_WIDTH; j < sq_dimension; j++) {
            sum += sq_matrix_1[row * sq_dimension + j] * sq_matrix_2[j * sq_dimension + col];
        }

        sq_matrix_result[row * sq_dimension + col] = sum;
    }
  }

  /* This function computes a matrix multiplication utilizing cuda for
     parallelism. */
  void matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2,
      float *sq_matrix_result, unsigned int sq_dimension) {

    float* sq_matrix_1_d;
    float* sq_matrix_2_d;
    float* sq_matrix_result_d;
    int size = sq_dimension * sq_dimension * sizeof(float);

    // Allocate device memory and copy inputs to the device
    cudaMalloc((void**) &sq_matrix_1_d, size);
    cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &sq_matrix_2_d, size);
    cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);

    // Allocate memory for the result on host
    cudaMalloc((void**) &sq_matrix_result_d, size);

    // Invoke the kernel to compute the matrix product
    int num_blocks = (sq_dimension - 1) / BLOCK_WIDTH + 1;
    int num_threads = min(BLOCK_WIDTH, sq_dimension);
    dim3 dim_grid(num_blocks, num_blocks, 1);
    dim3 dim_block(num_threads, num_threads, 1);
    matrix_mul_kernel<<<dim_grid, dim_block>>>
        (sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);

    // Transfer result from device to host
    cudaMemcpy(sq_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);
    cudaFree(sq_matrix_1_d);
    cudaFree(sq_matrix_2_d);
    cudaFree(sq_matrix_result_d);
  }

}