
///
/// This Kernel adds two Vectors A and B in C on GPU
/// using coalesced memory access.
/// 

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = blockStartIndex + (threadIdx.x);
    int i;

    for( i=0; i<N; ++i ){
        C[threadStartIndex+i*blockDim.x] = A[threadStartIndex+i*blockDim.x] + B[threadStartIndex+i*blockDim.x];
    }
}
