/*
Author: Claire Winogrodzki

Port of seven point stencil code from HIP to CUDA

Problem statement:

    Compute the Laplacian of a grid function u on a equidistantly spaced grid using a finite difference approximation

    f = \delta u

    Input parameters:

    ./laplacian <nx> <ny> <nz> <blk_x> <blk_y> <blk_z>

Based on AMD lab notes : Finite difference method – Laplacian part 1

https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-finite-difference-docs-laplacian_part1/
https://github.com/amd/amd-lab-notes/tree/release/finite-difference
*/

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include "kernel.cu"

using precision = double;
using namespace std;

// CUDA error check
#define CUDA_CHECK(stat)                                           \
{                                                                 \
    if(stat != cudaSuccess)                                        \
    {                                                             \
        std::cerr << "CUDA error: " << cudaGetErrorString(stat) <<  \
        " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(-1);                                                 \
    }                                                             \
}

template <typename T>
__global__ void test_function_kernel(T *u, int nx, int ny, int nz, T hx, T hy, T hz) { 

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // Exit if this thread is outside the boundary
    if (i >= nx ||
        j >= ny ||
        k >= nz)
        return;

    size_t pos = i + nx * (j +  ny * k);

    T c = 0.5;
    T x = i*hx;
    T y = j*hy;
    T z = k*hz;
    T Lx = nx*hx;
    T Ly = ny*hy;
    T Lz = nz*hz;
    u[pos] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz);
}

template <typename T>
void test_function(T *d_f, int nx, int ny, int nz, T hx, T hy, T hz) {

    dim3 block(256, 1);
    dim3 grid((nx - 1) / block.x + 1, ny, nz);

    test_function_kernel<<<grid, block>>>(d_f, nx, ny, nz, hx, hy, hz);
    CUDA_CHECK( cudaGetLastError() );
}

template <typename T>
int check(T *d_f, int nx, int ny, int nz, T hx, T hy, T hz, T tolerance=1e-6) {

    dim3 block(256, 1);
    dim3 grid((nx - 1) / block.x + 1, ny, nz);

    int *d_error; 
    CUDA_CHECK( cudaMalloc(&d_error, sizeof(int))   );
    CUDA_CHECK( cudaMemset(d_error, 0, sizeof(int)) );
    check_kernel<<<grid, block>>>(d_error, d_f, nx, ny, nz, hx, hy, hz, tolerance);
    CUDA_CHECK( cudaGetLastError() );
    int *error = new int[1];
    error[0] = 1;
    CUDA_CHECK( cudaMemcpy(error, d_error, sizeof(int), cudaMemcpyDeviceToHost) ); 
    int out = error[0];
    delete[] error;

    return out;
}

int main(int argc, char **argv){
  
    // Default thread block sizes
    int BLK_X = 256;
    int BLK_Y = 1;
    int BLK_Z = 1;

    // Default problem size
    size_t nx = 512, ny = 512, nz = 512; // Cube dimensions

    precision tolerance = 3e-6;

    int num_iter = 1000;
    #ifndef LB
    #define LB 1024

    cout << "nx,ny,nz = " << nx << ", " << ny << ", " << nz << endl;
    cout << "block sizes = " << BLK_X << ", " << BLK_Y << ", " << BLK_Z << endl;

    // Theoretical fetch and write sizes:
    size_t theoretical_fetch_size = (nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2) ) * sizeof(precision);
    size_t theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * sizeof(precision);

    std::cout << "Theoretical fetch size (GB): " << theoretical_fetch_size * 1e-9 << std::endl;
    std::cout << "Theoretical write size (GB): " << theoretical_write_size * 1e-9 << std::endl;

    size_t numbytes = nx * ny * nz * sizeof(precision);

    // Input and output arrays
    precision *d_u, *d_f; 

    // Allocate on device
    CUDA_CHECK( cudaMalloc((void**)&d_u, numbytes) );
    CUDA_CHECK( cudaMalloc((void**)&d_f, numbytes) );

    // Grid point spacings
    precision hx = 1.0 / (nx - 1), hy = 1.0 / (ny - 1), hz = 1.0 / (nz - 1); 
    
    // Initialize test function: 0.5 * (x * (x - 1) + y * (y - 1) + z * (z - 1))
    test_function(d_u, nx, ny, nz, hx, hy, hz);

    // Compute Laplacian (1/2) (x(x-1) + y(y-1) + z(z-1)) = 3 for all interior points
    laplacian(d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz);

    // Verification
    int error = check(d_f, nx, ny, nz, hx, hy, hz, tolerance);
    if (error)
        cout << "Correctness test failed. Pointwise error larger than " << tolerance << endl;

    // Timing
    float total_elapsed = 0;
    float elapsed;
    cudaEvent_t start, stop;
    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop)  );

    for (int iter = 0; iter < num_iter; ++iter) {
        // Flush cache
        CUDA_CHECK( cudaDeviceSynchronize()                     );
        CUDA_CHECK( cudaEventRecord(start)                      );
        laplacian(d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz);
        CUDA_CHECK( cudaGetLastError()                          );
        CUDA_CHECK( cudaEventRecord(stop)                       );
        CUDA_CHECK( cudaEventSynchronize(stop)                  );
        CUDA_CHECK( cudaEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }

    // Effective memory bandwidth
    size_t datasize = theoretical_fetch_size + theoretical_write_size;
    printf("Laplacian kernel took: %g ms, effective memory bandwidth: %g GB/s \n",
            total_elapsed / num_iter,
            datasize * num_iter / total_elapsed / 1e6
            );

    // Clean up
    CUDA_CHECK( cudaFree(d_f) );
    CUDA_CHECK( cudaFree(d_u) );

    return 0;
}