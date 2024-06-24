/******************************************************************************
Copyright (c) 2022 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
******************************************************************************/

/*
 Problem statement:

    Compute the Laplacian of a grid function u on a equidistantly spaced grid
 using a finite difference approximation

    f = \delta u

    Input parameters:

    ./laplacian <nx> <ny> <nz>
*/

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <vector>

#ifdef DOUBLE
using precision = double;
#else
using precision = float;
#endif

std::string getEnvVar(const std::string &key) {
  char *val = getenv(key.c_str());
  return val == NULL ? std::string("") : std::string(val);
}

/**
 * @brief Time in microseconds
 *
 * @param start
 * @param end
 * @return double
 */
static double dtime(struct timespec start, struct timespec end) {
  return ((double)((end.tv_sec - start.tv_sec) * 1000000 +
                   (end.tv_nsec - start.tv_nsec) / 1000)) /
         1E6;
}

template <class T>
void test_function(std::vector<T> &u, int nx, int ny, int nz, T hx, T hy,
                   T hz) {

  //   int i = threadIdx.x + blockIdx.x * blockDim.x;
  //   int j = threadIdx.y + blockIdx.y * blockDim.y;
  //   int k = threadIdx.z + blockIdx.z * blockDim.z;

  //   // Exit if this thread is outside the boundary
  //   if (i >= nx || j >= ny || k >= nz)
  //     return;

  T Lx = nx * hx;
  T Ly = ny * hy;
  T Lz = nz * hz;

  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int k = 1; k < nz - 1; k++) {
        size_t pos = i + nx * (j + ny * k);

        T c = 0.5;
        T x = i * hx;
        T y = j * hy;
        T z = k * hz;

        u[pos] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz);
      }
    }
  }
}

template <class T>
void laplacian(std::vector<T> &f, const std::vector<T> &u, int nx, int ny,
               int nz, T hx, T hy, T hz) {

  //   int i = threadIdx.x + blockIdx.x * blockDim.x;
  //   int j = threadIdx.y + blockIdx.y * blockDim.y;
  //   int k = threadIdx.z + blockIdx.z * blockDim.z;

  //   // Exit if this thread is on the boundary
  //   if (i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz -
  //   1)
  //     return;

  const T invhx2 = (T)1. / hx / hx;
  const T invhy2 = (T)1. / hy / hy;
  const T invhz2 = (T)1. / hz / hz;
  const T invhxyz2 = -2. * (invhx2 + invhy2 + invhz2);

  int i, j, k;
  size_t pos;
  const size_t slice = nx * ny;
#ifdef _OPENMP
#pragma omp parallel for default(shared) private(i, j, k, pos)
#endif
  for (i = 1; i < nx - 1; ++i) {
    for (j = 1; j < ny - 1; ++j) {
      for (k = 1; k < nz - 1; ++k) {
        pos = i + nx * j + nx * ny * k;
        // Compute the result of the stencil operation
        f[pos] = u[pos] * invhxyz2 + (u[pos - 1] + u[pos + 1]) * invhx2 +
                 (u[pos - nx] + u[pos + nx]) * invhy2 +
                 (u[pos - slice] + u[pos + slice]) * invhz2;
      }
    }
  }
}

int main(int argc, char **argv) {

  // Default problem size
  size_t nx = 512, ny = 512, nz = 512;
#ifdef DOUBLE
  precision tolerance = 3e-6;
#else
  precision tolerance = 3e-1;
#endif
  int num_iter = 1000;

  if (argc > 1)
    nx = atoi(argv[1]);
  if (argc > 2)
    ny = atoi(argv[2]);
  if (argc > 3)
    nz = atoi(argv[3]);

#ifdef DOUBLE
  std::cout << "Precision: double" << std::endl;
#else
  std::cout << "Precision: float" << std::endl;
#endif
  std::cout << "nx,ny,nz = " << nx << ", " << ny << ", " << nz << std::endl;

  // Theoretical fetch and write sizes:
  size_t theoretical_fetch_size =
      (nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)) *
      sizeof(precision);
  size_t theoretical_write_size =
      ((nx - 2) * (ny - 2) * (nz - 2)) * sizeof(precision);

#ifdef THEORY
  std::cout << "Theoretical fetch size (GB): " << theoretical_fetch_size * 1e-9
            << endl;
  std::cout << "Theoretical write size (GB): " << theoretical_write_size * 1e-9
            << endl;
#endif

  size_t numbytes = nx * ny * nz * sizeof(precision);

  std::vector<precision> d_u(nx * ny * nz);
  std::vector<precision> d_f(nx * ny * nz);

  // Grid spacings
  precision hx = 1.0 / (nx - 1);
  precision hy = 1.0 / (ny - 1);
  precision hz = 1.0 / (nz - 1);

  // Initialize test function: 0.5 * (x * (x - 1) + y * (y - 1) + z * (z - 1))
  test_function<precision>(d_u, nx, ny, nz, hx, hy, hz);

  // Compute Laplacian (1/2) (x(x-1) + y(y-1) + z(z-1)) = 3 for all interior
  // points

  laplacian<precision>(d_f, d_u, nx, ny, nz, hx, hy, hz);

  // Timing
  double total_elapsed = 0;
  struct timespec start_i, end_i;

  for (int iter = 0; iter < num_iter; ++iter) {
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_i);
    laplacian<precision>(d_f, d_u, nx, ny, nz, hx, hy, hz);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_i);
    const double elapsed = dtime(start_i, end_i);
    total_elapsed += elapsed;
  }

  std::cout << "Total time: " << total_elapsed << " s"
            << " kernel took: " << total_elapsed / num_iter * 1E3
            << " ms using " << getEnvVar("OMP_NUM_THREADS") << " threads"
            << std::endl;

  // Effective memory bandwidth
  //   size_t datasize = theoretical_fetch_size + theoretical_write_size;
  //   printf("Laplacian kernel took: %g ms, effective memory bandwidth: %g GB/s
  //   \n",
  //          total_elapsed / num_iter, datasize * num_iter / total_elapsed /
  //          1e6);

  return 0;
}