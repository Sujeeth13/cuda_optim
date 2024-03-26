# CUDA Optimization

## Commands to run for each part:
Part A:
    - make
        Q1:
            ./vecadd00 <values per thread>
        Q2:
            ./vecadd01 <values per thread>
        Q3:
            ./matmult00 < 16 or 32 or 64> (for 256,512,1024 sized matrix)
        Q4:
           ./matmult01 < 8or 16 or 32> (for 256,512,1024 sized matrix) 

Part B:
    Q1:
        g++ q1.cpp -o q1
        ./q1 < K >
    Q2:
        nvcc q2.cu -o q2
        ./q2 <K> <1 or 2 or 3> (second argument is to specify which kernel to run of the three scenarios)
    Q3:
        nvcc q3.cu -o q3
        ./q3 <K> <1 or 2 or 3> (second argument is to specify which kernel to run of the three scenarios)
    Q4:
        python3 q4.py

Part C:
    Q1:
        nvcc c1.cu -o cu
        ./c1
    Q2:
        nvcc c2.cu -o c2
        ./c2
    Q3:
        nvcc c3.cu -o c3 -lcudnn
        ./c3

