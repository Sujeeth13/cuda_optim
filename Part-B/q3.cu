#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

__global__ void add1x1(int N, const float* A,const float* B, float* C) {
    for (int i=0;i<N; i++)
        C[i] = A[i] + B[i];
}

__global__ void add1x256(int N, const float* A,const float* B, float* C) {
    int stride = blockDim.x;
    int id = threadIdx.x;

    for(int i=id; i<N; i+=stride)
        C[i] = A[i] + B[i];
}

__global__ void addNx256(int N, const float* A,const float* B, float* C) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i=id; i<N; i+=stride)
        C[i] = A[i] + B[i];
}

int main(int argc, char *argv[]) {
    int K=1;
    if(argc == 2) {
        K = atoi(argv[1]);
    }
    int N = K*1e6;

    printf("Size of vector: %d\n",N);

    struct timeval time;
    double start,end;
    float *A,*B,*C;

    if (cudaMallocManaged(&A,N*sizeof(float))!= cudaSuccess) {
        printf("Failed to allocate memory to A\n");
        exit(0);
    }
    if (cudaMallocManaged(&B,N*sizeof(float)) != cudaSuccess) {
        printf("Failed to allocate memory to B\n");
        exit(0);
    }
    if (cudaMallocManaged(&C,N*sizeof(float)) != cudaSuccess) {
        printf("Failed to allocate memory to C\n");
        exit(0);
    }

    // init the matrix
    for (int i=0;i<N;i++) {
        A[i] = (float)i;
        B[i] = (float)i;
        C[i] = 0;
    }
    // warm up
    add1x1<<<1,1>>>(N,A,B,C);
    add1x256<<<1,256>>>(N,A,B,C);
    addNx256<<<((N + 255)/256),256>>>(N,A,B,C);
    cudaDeviceSynchronize();

    // profiling of 1,1 kernel
    gettimeofday(&time, NULL);
    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    add1x1<<<1,1>>>(N,A,B,C);
    cudaDeviceSynchronize();

    gettimeofday(&time, NULL);
    end = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
    printf("Time taken by the 1x1 kernel: %lf\n",(end-start));

    // profiling of 1,256 kernel
    gettimeofday(&time, NULL);
    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    add1x256<<<1,256>>>(N,A,B,C);
    cudaDeviceSynchronize();

    gettimeofday(&time, NULL);
    end = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
    printf("Time taken by the 1x256 kernel: %lf\n",(end-start));

    // profiling of N,256 kernel
    gettimeofday(&time, NULL);
    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    addNx256<<<((N + 255)/256),256>>>(N,A,B,C);
    cudaDeviceSynchronize();

    gettimeofday(&time, NULL);
    end = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
    printf("Time taken by the Nx256 kernel: %lf\n",(end-start));

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}