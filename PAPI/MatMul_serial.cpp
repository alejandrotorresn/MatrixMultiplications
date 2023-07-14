/*
    argv[] ->   N: rows of A
    argv[] ->   M: columns of B
    argv[] ->   K: columns of A and rows of B
    argv[] ->   newMatrix: if 1 create new values of matrix A and B. If 0 read from file
    argv[] ->   iter: number if iterations of matrix multiplications
    argv[] ->   printOp: if 1 print.
    argv[] ->   times: if 1 print.
*/
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <papi.h>
#include "MatMul.h"

using namespace std;

static void matMul_serial(const float *A, const float *B, float *C, int N, int M, int K);
void handle_error(int retval);

int main(int argc, char **argv) {

    float *matA, *matB, *matC;
    int N, M, K;
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

    // Allocation memory space
    matA = (float *)malloc(M * K * sizeof(float));
    matB = (float *)malloc(K * N * sizeof(float));
    matC = (float *)malloc(M * N * sizeof(float));

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
    
    // unsync the I/O of C and C++
    ios_base::sync_with_stdio(false);

    // --------------------------------------------------------------------------------------------------------------
    
    // start timer
    auto t1 = chrono::steady_clock::now();

    retval = PAPI_hl_region_begin("computation");
    if (retval != PAPI_OK)
        handle_error(1);

    for (int i=0; i<iter; i++)
        matMul_serial(matA, matB, matC, N, M, K);
    
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


static void matMul_serial(const float *A, const float *B, float *C, int N, int M, int K) {
    for (int row=0; row<M; row++) {
        for (int col=0; col<N; col++) {
            float element_c = 0.f;
            for (int e=0; e<K; e++) {
                element_c += A[row * K + e] * B[e * N + col];
            }
            C[row * N + col] = element_c;
        }
    }
}