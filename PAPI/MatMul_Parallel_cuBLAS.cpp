/*
    argv[] ->   N: rows of A
    argv[] ->   M: columns of B
    argv[] ->   K: columns of A and rows of B
    argv[] ->   newMatrix: if 1 create new values of matrix A and B. If 0 read from file
    argv[] ->   iter: number if iterations of matrix multiplications
    argv[] ->   printOp: if 1 print.
*/
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <papi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "MatMul.h"

using namespace std;

int main(int argc, char **argv) {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaStream_t stream;

    float *matA, *matB, *matC;
    float *d_A, *d_B, *d_C;
    int N, M, K;
    float alpha, beta;
    int newMatrix;
    int iter;
    int printOp;
    int times;

    int retval;
    
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);
    newMatrix = atoi(argv[4]);
    iter = atoi(argv[5]);
    printOp = atoi(argv[6]);
    times = atoi(argv[7]);

    alpha = 1.f;
    beta = 1.f;

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

    /// --------------------------------------------------------------------------------------------------------------
    
    // start timer
    auto t1 = chrono::steady_clock::now();

    retval = PAPI_hl_region_begin("computation");
    if (retval != PAPI_OK)
        handle_error(1);

    for (size_t i=0; i<iter; i++) {

        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            cout << "CUBLAS initialization failed" << endl;
            return EXIT_FAILURE;
        }


        // allocation memory space
        cudaMalloc((void **)&d_A, M * K * sizeof(float));
        cudaMalloc((void **)&d_B, K * N * sizeof(float));
        cudaMalloc((void **)&d_C, M * N * sizeof(float));

        cudaStat = cudaStreamCreate(&stream);

        cublasSetMatrixAsync(M, K, sizeof(*d_A), matA, M, d_A, M, stream);
        cublasSetMatrixAsync(K, N, sizeof(*d_B), matB, K, d_B, K, stream);
        cublasSetMatrixAsync(M, N, sizeof(*d_C), matC, M, d_C, M, stream);

        cublasSetStream(handle, stream);
    
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
        
        cublasGetMatrixAsync(M, N, sizeof(*d_C), d_C, M, matC, M, stream);
        cudaStreamSynchronize(stream);

        cublasDestroy(handle);
        cudaStreamDestroy(stream);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    
    }

    retval = PAPI_hl_region_end("computation");
    if (retval != PAPI_OK)
        handle_error(1);

    // end timer
    auto t2 = chrono::steady_clock::now();

    // --------------------------------------------------------------------------------------------------------------
    
    
    // Calculating total time
    if (times == 1)
        cout << (double)chrono::duration_cast<chrono::microseconds>(t2 - t1).count()/1000000.0f/iter << " ";


    if (printOp == 1)
        printMat(matC, N, M);

    free(matA);
    free(matB);
    free(matC);

    return 0;
}