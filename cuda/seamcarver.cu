//
// 18645 - GPU Seamcarving
// Authors: Adu Bhandaru, Matt Sarett
//

#include <cuda.h>
#include <cuda_runtime.h>
#include "seamcarver.h"

#define BLOCK_WIDTH 1024

using std::cout;
using std::endl;
using std::min;
using std::vector;


Seamcarver::Seamcarver(Image* image) {
  cout << ">> init seamcarver" << endl;
  _image = image;
}


Seamcarver::~Seamcarver() {

}


// Simply remove n seams.
void Seamcarver::removeSeams(int n) {
  cout << "   removing " << n << " seams ..." << endl;
  for (int i = 0; i < n; i++) {
    removeSeam();
  }
}


// Removes 1 seam.
void Seamcarver::removeSeam() {
  findSeam();
  _image->removeSeam(_seam);
}


__global__ void compute_min_cost_kernel(float* energies, float* min_costs,
    int width, int height) {

  // Extract thread and block index information
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int col = bx * BLOCK_WIDTH + tx;

  // Allocate shared memory with padding to avoid bank conflicts
  __shared__ float shared_costs[BLOCK_WIDTH];

  // Load the first row of shared memory with energies and min costs
  if (col < width) {
    shared_costs[tx] = energies[col];
    min_costs[col] = energies[col];
  }
  else {
    return;
  }

  // Wait for all threads to finish loading the first row of shared memory
  __syncthreads();

  for (int row = 1; row < height; row++) {
    float left;
    float middle = shared_costs[tx];
    float right;

    if (tx > 0) {
      left = shared_costs[tx - 1];
    }
    else {
      left = MAX_VALUE;
    }

    if (tx + 1 < width) {
      right = shared_costs[tx + 1];
    }
    else {
      right = MAX_VALUE;
    }

    // Compute the minimum and then add cost of current cell
    float minimum = min(left, min(middle, right));
    float cost = minimum + energies[row * width + col];
    __syncthreads();
    shared_costs[tx] = cost;
    __syncthreads();
    min_costs[row * width + col] = cost;
  }
}


// Finds the seam of the lowest cost.
void Seamcarver::findSeam() {
  Energies energies(_image);
  energies.compute();
  float* energies_h = energies.getEnergies();

  // Declare pointers for device and host memory
  float* energies_d;
  float* min_cost_d;
  int width = energies.width();
  int height = energies.height();
  int size = width * height * sizeof(float);

  // Allocate device memory and for inputs and outputs
  cudaMalloc((void**) &energies_d, size);
  cudaMemcpy(energies_d, energies_h, size, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &min_cost_d, size);

  // Invoke the kernel to compute the energies
  int num_blocks = (width - 1) / BLOCK_WIDTH + 1;
  int num_threads = min(BLOCK_WIDTH, width);
  dim3 dim_grid(num_blocks, 1, 1);
  dim3 dim_block(num_threads, 1, 1);
  compute_min_cost_kernel<<<dim_grid, dim_block>>>
      (energies_d, min_cost_d, width, height);

  // Transfer result from device to host
  cudaMemcpy(energies_h, min_cost_d, size, cudaMemcpyDeviceToHost);
  cudaFree(energies_d);
  cudaFree(min_cost_d);

  // Find the minimum value in the bottom row.
  int min_index = 0;
  for (int col = 0; col < width; col++) {
    float min = energies.get(height - 1, min_index);
    float test = energies.get(height - 1, col);
    if (test < min) {
      min_index = col;
    }
  }

  // Create the seam in reverse order.
  _seam.clear();
  _seam.push_back(min_index);
  for (int i = height - 2; i >= 0; i--) {
    float left = energies.get(i, min_index - 1);
    float middle = energies.get(i, min_index);
    float right = energies.get(i, min_index + 1);

    // Have the seam follow the least cost.
    if (left < middle && left < right) {
      min_index--; // go left
    } else if (right < middle && right < left) {
      min_index++; // go right
    }

    // Append to the seam.
    _seam.push_back(min_index);
  }

  // Clean up.
  std::reverse(_seam.begin(), _seam.end());
}


float Seamcarver::minCost(Energies& energies, int i, int j) {
  // For top row entries we return the element itself.
  if (i <= 0) {
    return energies.get(i, j);
  }

  // Take the 3 adjacent cells in the above row.
  float left = energies.get(i - 1, j - 1);
  float middle = energies.get(i - 1, j);
  float right = energies.get(i - 1, j + 1);

  // Compute the minimum, add cost of current cell.
  float minimum = min(left, min(middle, right));
  return minimum + energies.get(i, j);
}
