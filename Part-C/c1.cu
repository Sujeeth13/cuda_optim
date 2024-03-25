#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

__global__ void Conv2d(const double *I,const double *F,double* O,int C,int K, int H,int W,int HP,int WP,int FH, int FW)
{
    int k = blockIdx.z;
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;
    int thread_row = threadIdx.x;
    int thread_col = threadIdx.y;

    double OValue = 0;

    for(int c=0; c<C; ++c) {
        for(int i = 0; i<FH; ++i) {
            for(int j=0; j<FW; ++j) {
                OValue += I[c*(HP*WP) + block_row*WP*BLOCK_SIZE + block_col*BLOCK_SIZE + (thread_row+i)*WP + thread_col+j] * F[k*(C*FH*FW) + c*(FH*FW) + (FH-1-i)*FW + (FW-1-j)];
            }
        }
    }

    O[k*(H*W) + block_row*W*BLOCK_SIZE + block_col*BLOCK_SIZE + thread_row*W + thread_col] = OValue;
}   

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

int main(int argc,char *argv[]) {
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
    // kernel code
    dim3 gridDim((H + BLOCK_SIZE -1)/BLOCK_SIZE, (W + BLOCK_SIZE - 1)/BLOCK_SIZE,K);
    // dim3 gridDim(256,256);
    // dim3 blockDim(4,4,K);
    dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);

    gettimeofday(&time, NULL);
    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    Conv2d<<<gridDim,blockDim>>>(d_I,d_F,d_O,C,K,H,W,HP,WP,FH,FW);
    cudaDeviceSynchronize();

    gettimeofday(&time, NULL);
    end = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Error: %s\n",cudaGetErrorString(err));
        exit(0);
    }

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
    return 0;
}