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
    float *h_A,*h_B,*h_C;
    float *d_A,*d_B,*d_C;
    h_A = (float*)malloc(N*sizeof(float));
    h_B = (float*)malloc(N*sizeof(float));
    h_C = (float*)malloc(N*sizeof(float));

    // init the matrix
    for (int i=0;i<N;i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)i;
    }

    if (cudaMalloc((void**)&d_A,N*sizeof(float)) != cudaSuccess) {
        printf("Failed to allocate GPU memory to d_A\n");
        exit(0);
    }
    if (cudaMalloc((void**)&d_B,N*sizeof(float)) != cudaSuccess) {
        printf("Failed to allocate GPU memory to d_B\n");
        exit(0);
    }
    if (cudaMalloc((void**)&d_C,N*sizeof(float)) != cudaSuccess) {
        printf("Failed to allocate GPU memory to d_C\n");
        exit(0);
    }
    if (cudaMemcpy(d_A,h_A,N*sizeof(float),cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Failed to copy vector A from host to device\n");
        exit(0);
    }
    if (cudaMemcpy(d_B,h_B,N*sizeof(float),cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Failed to copy vector B from host to device\n");
        exit(0);
    }
    // warm up
    add1x1<<<1,1>>>(N,d_A,d_B,d_C);
    add1x256<<<1,256>>>(N,d_A,d_B,d_C);
    addNx256<<<((N + 255)/256),256>>>(N,d_A,d_B,d_C);

    cudaDeviceSynchronize();

    // profiling of 1,1 kernel
    gettimeofday(&time, NULL);
    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    add1x1<<<1,1>>>(N,d_A,d_B,d_C);
    if (cudaMemcpy(h_C,d_C,N*sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Failed to copy vector A from host to device\n");
        exit(0);
    }
    cudaDeviceSynchronize();

    gettimeofday(&time, NULL);
    end = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
    printf("Time taken by the 1x1 kernel: %lf\n",(end-start));

    // profiling of 1,256 kernel
    gettimeofday(&time, NULL);
    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    add1x256<<<1,256>>>(N,d_A,d_B,d_C);
    if (cudaMemcpy(h_C,d_C,N*sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Failed to copy vector A from host to device\n");
        exit(0);
    }
    cudaDeviceSynchronize();

    gettimeofday(&time, NULL);
    end = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
    printf("Time taken by the 1x256 kernel: %lf\n",(end-start));

    // profiling of N,256 kernel
    gettimeofday(&time, NULL);
    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    addNx256<<<((N + 255)/256),256>>>(N,d_A,d_B,d_C);
    if (cudaMemcpy(h_C,d_C,N*sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Failed to copy vector A from host to device\n");
        exit(0);
    }
    cudaDeviceSynchronize();
    
    gettimeofday(&time, NULL);
    end = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
    printf("Time taken by the Nx256 kernel: %lf\n",(end-start));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}