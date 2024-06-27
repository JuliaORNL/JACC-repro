/*
Author: Claire Winogrodzki

HIP (AMD) version of seven point stencil

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
#include "kernel.hpp"

using precision = double;
using namespace std;

// HIP error check
#define HIP_CHECK(stat)                                           \
{                                                                 \
    if(stat != hipSuccess)                                        \
    {                                                             \
        std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
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
    HIP_CHECK( hipGetLastError() );
}


template <typename T>
__global__ void check_kernel(int *error, const T *f, 
                             int nx, int ny, int nz, 
                             T hx, T hy, T hz,
                             double tolerance) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // Exit if this thread is on the boundary
    if (i == 0 || i >= nx - 1 ||
        j == 0 || j >= ny - 1 ||
        k == 0 || k >= nz - 1)
        return;

    size_t pos = i + nx * (j + ny * k);

    // If the pointwise error exceeds the tolerance, we signal that an error has occurred

    // Laplacian of u when u is initialized using `test_function_kernel`
    T expected_f = 3;
    if ( fabs(f[pos] - expected_f) / expected_f > tolerance) {
        atomicAdd(error, 1); 
    }
} 


template <typename T>
int check(T *d_f, int nx, int ny, int nz, T hx, T hy, T hz, T tolerance=1e-6) {

    dim3 block(256, 1);
    dim3 grid((nx - 1) / block.x + 1, ny, nz);

    int *d_error; 
    HIP_CHECK( hipMalloc(&d_error, sizeof(int))   );
    HIP_CHECK( hipMemset(d_error, 0, sizeof(int)) );
    check_kernel<<<grid, block>>>(d_error, d_f, nx, ny, nz, hx, hy, hz, tolerance);
    HIP_CHECK( hipGetLastError() );
    int *error = new int[1];
    error[0] = 1;
    HIP_CHECK( hipMemcpy(error, d_error, sizeof(int), hipMemcpyDeviceToHost) ); 
    int out = error[0];
    delete[] error;

    return out;
} 

int main(int argc, char **argv)
{
    // Default thread block sizes
    int BLK_X = 256;
    int BLK_Y = 1;
    int BLK_Z = 1;
    
    // Default problem size
    size_t nx = 512, ny = 512, nz = 512; // Cube dimensions

    precision tolerance = 3e-6;

    int num_iter = 1000;

    if (argc > 1) nx = atoi(argv[1]);
    if (argc > 2) ny = atoi(argv[2]);
    if (argc > 3) nz = atoi(argv[3]);
    if (argc > 4) BLK_X = atoi(argv[4]);
    if (argc > 5) BLK_Y = atoi(argv[5]);
    if (argc > 6) BLK_Z = atoi(argv[6]);

    // #define LB 1024
    //     if (BLK_X * BLK_Y * BLK_Z > LB) {
    //         cout << "WARNING: input thread block size " << BLK_X <<"*" << BLK_Y << "*" << BLK_Z;
    //         cout << " exceeds limit " << LB << ", resetting to " << LB << "*1*1" << endl;
    //         BLK_X = LB;
    //         BLK_Y = 1;
    //         BLK_Z = 1;
    //     }

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
    HIP_CHECK( hipMalloc((void**)&d_u, numbytes) );
    HIP_CHECK( hipMalloc((void**)&d_f, numbytes) );

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
    hipEvent_t start, stop;
    HIP_CHECK( hipEventCreate(&start) );
    HIP_CHECK( hipEventCreate(&stop)  );

    for (int iter = 0; iter < num_iter; ++iter) {
        // Flush cache
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        laplacian(d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz);
        HIP_CHECK( hipGetLastError()                          );
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        total_elapsed += elapsed;
    }

    // Effective memory bandwidth
    size_t datasize = theoretical_fetch_size + theoretical_write_size;
    printf("Laplacian kernel took: %g ms, effective memory bandwidth: %g GB/s \n",
            total_elapsed / num_iter,
            datasize * num_iter / total_elapsed / 1e6
            );

    // Clean up
    HIP_CHECK( hipFree(d_f) );
    HIP_CHECK( hipFree(d_u) );

    return 0;
}