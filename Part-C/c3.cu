#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double getCheckSum(const double *h_O, int K, int H, int W) {
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                sum += h_O[k * H * W + i * W + j];
            }
        }
    }
    return sum;
}

int main(int argc, char *argv[]) {
    struct timeval time;
    double start,end;

    int H=1024,W=1024,C=3,FW=3,FH=3,K = 64;
    int P = 1;
    int HP = H + 2*P;
    int WP = W + 2*P;

    double *I,*F,*O;
    double *d_I,*d_F,*d_O;
    int size = C*(H + 2*P)*(W + 2*P);
    int o_size = K*H*W;
    int k_size = K*C*FH*FW;
    I = (double*)malloc(size*sizeof(double));
    O = (double*)malloc(o_size*sizeof(double));
    F = (double*)malloc(k_size*sizeof(double));

    // init matrix
    for(int c=0; c<C; ++c) {
        for(int x=0; x<HP; ++x) {
            for(int y=0; y<WP; ++y) {
                if (x == 0 || x == HP-1 || y==0 || y == WP-1)
                    I[c*(HP*WP) + x*WP + y] = 0;
                else
                    I[c*(HP*WP) + x*WP + y] = c*(x+y);
            }
        }
    }

    //init kernel
    for(int k=0; k<K; ++k) {
        for(int c=0; c<C; ++c) {
            for(int x=0; x<FH; ++x) {
                for(int y=0; y<FW; ++y) {
                    F[k*(C*FH*FW) + c*(FH*FW) + x*FW + y] = (c+k)*(x+y);
                }
            }
        }
    }

    if (cudaMalloc((void**)&d_I,size*sizeof(double)) != cudaSuccess) {
        printf("Failed to allocate GPU memory to I\n");
        exit(0);
    }
    if(cudaMalloc((void**)&d_O,o_size*sizeof(double)) != cudaSuccess) {
        printf("Failed to allocate GPU memory to O\n");
        exit(0);
    }
    if(cudaMalloc((void**)&d_F,k_size*sizeof(double)) != cudaSuccess) {
        printf("Failed to allocate GPU memory to F\n");
        exit(0);
    }
    if(cudaMemcpy(d_I,I,size*sizeof(double),cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Failed to copy I from host to device\n");
        exit(0);
    }
    if(cudaMemcpy(d_F,F,k_size*sizeof(double),cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Failed to copy F from host to device\n");
        exit(0);
    }

    cudnnHandle_t cudnn;
    if(cudnnCreate(&cudnn) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to create handle\n");
        exit(0);
    }
    cudnnTensorDescriptor_t inp;
    if(cudnnCreateTensorDescriptor(&inp) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to create input tensor descriptor\n");
        exit(0);
    }
    if(cudnnSetTensor4dDescriptor(inp,CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, HP, WP) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to set input tensor\n");
        exit(1);
    }

    cudnnFilterDescriptor_t filter;
    if(cudnnCreateFilterDescriptor(&filter) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to create filter tensor descriptor\n");
        exit(0);
    }
    if(cudnnSetFilter4dDescriptor(filter,CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to set filter tensor\n");
        exit(1);
    }

    cudnnTensorDescriptor_t out;
    if(cudnnCreateTensorDescriptor(&out) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to create output tensor descriptor\n");
        exit(0);
    }
    if(cudnnSetTensor4dDescriptor(out,CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to set output tensor\n");
        exit(1);
    }

    cudnnConvolutionDescriptor_t convolutionDescriptor;
    if (cudnnCreateConvolutionDescriptor(&convolutionDescriptor) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to create convolution descriptor\n");
        exit(1);
    }
    if (cudnnSetConvolution2dDescriptor(convolutionDescriptor, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to set convolution descriptor\n");
        exit(1);
    }

    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    int returnedAlgoCount;

    if (cudnnFindConvolutionForwardAlgorithm(cudnn, inp, filter, convolutionDescriptor, out, 1, &returnedAlgoCount, &algoPerf) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to get convolution algorithm\n");
        exit(1);
    }
    cudnnConvolutionFwdAlgo_t convolutionAlgorithm = algoPerf.algo;

    size_t workspaceBytes = 0;
    if (cudnnGetConvolutionForwardWorkspaceSize(cudnn, inp, filter, convolutionDescriptor, out, convolutionAlgorithm, &workspaceBytes) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to get convolution algorithm workspace\n");
        exit(1);
    }
    void* d_workspace = nullptr;
    if (cudaMalloc(&d_workspace, workspaceBytes) != cudaSuccess) {
        printf("Error: Unable to allocate workspace memory\n");
        exit(1);
    }

    gettimeofday(&time, NULL);
    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    const double alpha = 1.0, beta = 0.0;
    if (cudnnConvolutionForward(cudnn, &alpha, inp, d_I, filter, d_F, convolutionDescriptor, convolutionAlgorithm, d_workspace, workspaceBytes, &beta, out, d_O) != CUDNN_STATUS_SUCCESS) {
        printf("Error: Unable to compute convolution\n");
        exit(1);
    }

    gettimeofday(&time, NULL);
    end = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    if(cudaMemcpy(O,d_O,o_size*sizeof(double),cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Failed to copy O from device to host\n");
        exit(0);
    }

    double checksum = getCheckSum(O,K,H,W);
    printf("%.3lf,%.3lf\n",checksum,(end-start)*1e3);

    cudaFree(d_I);
    cudaFree(d_O);
    cudaFree(d_F);

    free(I);
    free(O);
    free(F);

    cudaFree(d_workspace);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    cudnnDestroyFilterDescriptor(filter);
    cudnnDestroyTensorDescriptor(out);
    cudnnDestroyTensorDescriptor(inp);
    cudnnDestroy(cudnn);
    return 0;
}
