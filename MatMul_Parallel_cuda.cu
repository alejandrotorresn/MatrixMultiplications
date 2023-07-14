/*
    argv[] ->   N: rows of A
    argv[] ->   M: columns of B
    argv[] ->   K: columns of A and rows of B
    argv[] ->   newMatrix: if 1 create new values of matrix A and B. If 0 read from file
    argv[] ->   iter: number if iterations of matrix multiplications
    argv[] ->   prsize_tOp: if 1 prsize_t.
*/
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include "MatMul.h"

#define BLOCK_DIM 16

using namespace std;

__global__ void matMul_cuda_kernel(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);
bool verification(const float *C_cpu, const float *C_gpu, size_t length);


int main(int argc, char **argv) {

    float *matA, *matB, *matC;
    //float *matC_cpu;
    float *d_A, *d_B, *d_C;
    size_t N, M, K;
    size_t newMatrix;
    size_t iter;
    size_t prsize_tOp;
    int times;

    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);
    newMatrix = atoi(argv[4]);
    iter = atoi(argv[5]);
    prsize_tOp = atoi(argv[6]);
    times = atoi(argv[7]);

    // Allocation memory space
    matA = (float *)malloc(M * K * sizeof(float));
    matB = (float *)malloc(K * N * sizeof(float));
    matC = (float *)malloc(M * N * sizeof(float));
    //matC_cpu = (float *)malloc(M * N * sizeof(float));

    if ( newMatrix == 1 ) {
        // Create randomized values and save them to file
        random_init(matA, M * K, true);
        saveMat("matrixA.dat", matA, M * K);
        random_init(matB, K * N, true);
        saveMat("matrixB.dat", matB, K * N);
    } else if (newMatrix == 0) {
        // Read matrices from file
        readMat("matrixA.dat", matA, N*K);
        readMat("matrixB.dat", matB, K*M); 
    }

    random_init(matC, M * N, false);

    // unsync the I/O of C and C++
    ios_base::sync_with_stdio(false);

    const dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    const dim3 gridDim((N + BLOCK_DIM - 1)/BLOCK_DIM, (M + BLOCK_DIM - 1)/BLOCK_DIM);

    /// --------------------------------------------------------------------------------------------------------------
    
    // start timer
    auto t1 = chrono::steady_clock::now();

    for (size_t i=0; i<iter; i++) {

        // allocation memory space
        cudaMalloc((void **)&d_A, M * K * sizeof(float));
        cudaMalloc((void **)&d_B, K * N * sizeof(float));
        cudaMalloc((void **)&d_C, M * N * sizeof(float));

        // copy initial value for gpu  memory
        cudaMemcpy(d_A, matA, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, matB, K * N * sizeof(float), cudaMemcpyHostToDevice);

        matMul_cuda_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    
        // copy data from the gpu
        cudaMemcpy(matC, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    // end timer
    auto t2 = chrono::steady_clock::now();

    // --------------------------------------------------------------------------------------------------------------
    
    
    // Calculating total time
    if (times == 1)
        cout << (double)chrono::duration_cast<chrono::microseconds>(t2 - t1).count()/1000000.0f/iter << " ";

    if (prsize_tOp == 1)
        printMat(matC, N, M);

    free(matA);
    free(matB);
    free(matC);
    //free(matC_cpu);

    return 0;
}


__global__ void matMul_cuda_kernel(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {
    size_t bid_x = blockIdx.x; //* blockDim.x;
    size_t bid_y = blockIdx.y; //* blockDim.y;
    size_t tid_x = threadIdx.x;
    size_t tid_y = threadIdx.y;

    float element_c = 0.f;

    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

    size_t aBegin = K * BLOCK_DIM * bid_y;
    size_t aEnd = aBegin + K - 1;
    size_t aStep = BLOCK_DIM;

    size_t bBegin = BLOCK_DIM * bid_x;
    size_t bStep = BLOCK_DIM * N;

    for (size_t i=aBegin, j=bBegin; i<=aEnd; i+=aStep, j+=bStep) {
        s_tile_A[tid_y][tid_x] = A[i + K * tid_y + tid_x];
        s_tile_B[tid_y][tid_x] = B[j + N * tid_y + tid_x];

        __syncthreads();

        for (size_t k=0; k<BLOCK_DIM; ++k) {
            element_c += s_tile_A[tid_y][k] * s_tile_B[k][tid_x];
        }

        __syncthreads();
    }
    size_t cIdx = N * BLOCK_DIM * bid_y + BLOCK_DIM * bid_x;

    C[cIdx + N * tid_y + tid_x] = element_c;
}