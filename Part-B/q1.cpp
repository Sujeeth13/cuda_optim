#include <iostream>
#include <cstdlib>
#include <chrono>
#include <sys/time.h>

void add(const int N,const float* A, const float* B, float*& C) 
{
    for(int i=0;i<N;i++) {
        C[i] = A[i] + B[i];
    }
}
int main(int argc, char* argv[]) {
    int K = 1;
    if (argc == 2) {
        K = std::stoi(argv[1]);
    }
    struct timeval time;
    double start,end;
    float *A, *B, *C;
    int N = K*1e6;
    A = (float*)malloc(N*sizeof(float));
    B = (float*)malloc(N*sizeof(float));
    C = (float*)malloc(N*sizeof(float));

    std::cout<<"Size of vector: "<<N<<std::endl;
    /* Init the vector */
    for(int i=0; i<N; i++) {
        A[i] = i;
        B[i] = i;
    }

    // Profiling begins
    gettimeofday(&time, NULL);
    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
    
    add(N, A, B, C);

    gettimeofday(&time, NULL);
    end = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    std::cout << "Time taken for addition: " << (end - start) << " s" << std::endl;

    free(A);
    free(B);
    return 0;
}