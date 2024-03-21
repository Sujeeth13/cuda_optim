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

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C){
    float *A_sub,*B_sub,*C_sub; // stores the block values of the matrix

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    C_sub = &C.elements[C.width * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

    float CValue[S*S];
  #pragma unroll
    for (int s=0; s<S*S; s++)
      CValue[s] = 0;

    for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m) {
      
      A_sub = &A.elements[A.width * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
      B_sub = &B.elements[B.width * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

      __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
      __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

#pragma unroll
      for (int s=0; s< S*S ; ++s) {
        shared_A[thread_row][thread_col*S*S+s] = A_sub[thread_row*A.width + thread_col*S*S + s];
        shared_B[thread_row][thread_col*S*S+s] = B_sub[thread_row*B.width + thread_col*S*S + s];
      }
      
      __syncthreads();

      for (int s = 0; s < S*S; ++s) {
        for (int e=0; e<FOOTPRINT_SIZE; ++e) {
          CValue[s] += shared_A[thread_row][e] * shared_B[e][thread_col*S*S + s];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for(int s=0; s<S*S; ++s) {
      C_sub[thread_row*C.width + thread_col*S*S + s] = CValue[s];
    } 
}