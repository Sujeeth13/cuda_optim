#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
#define S FOOTPRINT_SIZE/BLOCK_SIZE

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

    // Each thread handles 2 rows and 2 columns
    int thread_row = threadIdx.y * 2;
    int thread_col = threadIdx.x * 2;  
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    float *Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];
    float Cvalues[4] = {0.0, 0.0, 0.0, 0.0};  // To hold the computed values

    for (int m = 0; m < (A.width / FOOTPRINT_SIZE); ++m) {
        float *Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
        float *Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

        __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
        __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

        for(int i = 0; i < S; ++i) {  // Load 2 rows
            for(int j = 0; j < S; ++j) {  // And 2 columns per thread
                shared_A[thread_row + i][thread_col + j] = Asub[(thread_row + i) * A.stride + (thread_col + j)];
                shared_B[thread_row + i][thread_col + j] = Bsub[(thread_row + i) * B.stride + (thread_col + j)];
            }
        }

        __syncthreads();

        // Compute 4 elements of C per thread
        for(int y = 0; y < S; ++y) {
            for(int x = 0; x < S; ++x) {
#pragma unroll
                for(int e = 0; e < FOOTPRINT_SIZE; ++e) {
                    Cvalues[y * S + x] += shared_A[thread_row + y][e] * shared_B[e][thread_col + x];
                }
            }
        }

        __syncthreads();
    }

    // Write the 4 computed values back to global memory
    for(int i = 0; i < S; ++i) {
        for(int j = 0; j < S; ++j) {
            Csub[(thread_row + i) * C.stride + thread_col + j] = Cvalues[i * S + j];
        }
    }
}
