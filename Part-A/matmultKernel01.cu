///
/// matmultKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
#define S FOOTPRINT_SIZE/BLOCK_SIZE

// __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
//     float *A_sub,*B_sub,*C_sub; // stores the block values of the matrix

//     int thread_row = threadIdx.y;
//     int thread_col = threadIdx.x;
//     int block_row = blockIdx.y;
//     int block_col = blockIdx.x;

//     C_sub = &C.elements[C.width * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

//     float CValue[S*S];
//   #pragma unroll
//     for (int s=0; s<S*S; s++)
//       CValue[s] = 0;

//     for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m) {
      
//       A_sub = &A.elements[A.width * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
//       B_sub = &B.elements[B.width * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

//       __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
//       __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

// #pragma unroll
//       for (int s=0; s< S*S ; ++s) {
//         // shared_A[S*thread_row*FOOTPRINT_SIZE + thread_col*S*S + s] = A_sub[S*thread_row*A.width + thread_col*S*S + s];
//         // shared_B[S*thread_row*FOOTPRINT_SIZE + thread_col*S*S + s] = B_sub[S*thread_row*B.width + thread_col*S*S + s];
//         shared_A[thread_col/(BLOCK_SIZE/S) + S*thread_row][((thread_col*S*S)%(FOOTPRINT_SIZE/(S*S))) + s] = A_sub[S*thread_row*A.width + thread_col*S*S + s];
//         shared_B[thread_col/(BLOCK_SIZE/S) + S*thread_row][((thread_col*S*S)%(FOOTPRINT_SIZE/(S*S))) + s] = B_sub[S*thread_row*A.width + thread_col*S*S + s];

//       }
      
//       __syncthreads();

//       for (int s=0; s<S*S; ++s) {
//         for (int e=0; e<FOOTPRINT_SIZE; ++e) {
//           CValue[s] += shared_A[thread_col/(BLOCK_SIZE/S) + S*thread_row][e] * shared_B[e][((thread_col*S*S)%(FOOTPRINT_SIZE/(S*S))) + s];
//         }
//       }
//     }
//     __syncthreads();

// #pragma unroll
//     for(int s=0; s<S*S; ++s) {
//       C_sub[S*thread_row*C.width + thread_col*S*S + s] = CValue[s];
//     } 
// }

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

    int thread_row = threadIdx.y * 2;  // Each thread handles 2 rows now
    int thread_col = threadIdx.x * 2;  // And 2 columns per iteration
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Adjusting for the new FOOTPRINT_SIZE
    float *Csub = &C.elements[C.stride * 32 * block_row + 32 * block_col];
    float Cvalues[4] = {0.0, 0.0, 0.0, 0.0};  // To hold the computed values

    for (int m = 0; m < (A.width / 32); ++m) {
        float *Asub = &A.elements[A.stride * 32 * block_row + 32 * m];
        float *Bsub = &B.elements[B.stride * 32 * m + 32 * block_col];

        __shared__ float shared_A[32][32];  // Adjusted for FOOTPRINT_SIZE = 32
        __shared__ float shared_B[32][32];  // Adjusted for FOOTPRINT_SIZE = 32
        // Load A_sub and B_sub into shared memory, ensuring coalesced access
        for(int i = 0; i < 2; ++i) {  // Load 2 rows
            for(int j = 0; j < 2; ++j) {  // And 2 columns per thread
                shared_A[thread_row + i][thread_col + j] = Asub[(thread_row + i) * A.stride + (thread_col + j)];
                shared_B[thread_row + i][thread_col + j] = Bsub[(thread_row + i) * B.stride + (thread_col + j)];
            }
        }

        __syncthreads();

        // Compute 4 elements of C per thread
        for(int y = 0; y < 2; ++y) {
            for(int x = 0; x < 2; ++x) {
#pragma unroll
                for(int e = 0; e < 32; ++e) {
                    Cvalues[y * 2 + x] += shared_A[thread_row + y][e] * shared_B[e][thread_col + x];
                }
            }
        }

        __syncthreads();
    }

    // Write the 4 computed values back to global memory
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            Csub[(thread_row + i) * C.stride + thread_col + j] = Cvalues[i * 2 + j];
        }
    }
}
