#include <iostream>
#include <cstdlib>
#include <chrono>

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

    /* warm up */
    add(N,A,B,C);

    // Profiling begins
    auto start = std::chrono::high_resolution_clock::now();
    
    add(N, A, B, C);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time taken for addition: " << (float)duration/1e6 << " s" << std::endl;

    free(A);
    free(B);
    return 0;
}