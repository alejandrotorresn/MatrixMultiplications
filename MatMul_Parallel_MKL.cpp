#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include "MatMul.h"

using namespace std;


int main(int argc, char **argv) {

    double *matA, *matB, *matC;
    int N, M, K;
    int newMatrix;
    int iter;
    int printOp;
    int times;

    N = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);
    newMatrix = atoi(argv[4]);
    iter = atoi(argv[5]);
    printOp = atoi(argv[6]);
    times = atoi(argv[7]);

    int i, j;
    double alpha, beta;
    
    alpha = 1.0;
    beta = 0.0;
    
    matA = (double *)mkl_malloc(M*K*sizeof(double), 64);
    matB = (double *)mkl_malloc(K*N*sizeof(double), 64);
    matC = (double *)mkl_malloc(M*N*sizeof(double), 64);
    
    if (matA == NULL || matB == NULL || matC == NULL) {
        printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(matA);
        mkl_free(matB);
        mkl_free(matC);
        return 1;
    }

    if ( newMatrix == 1 ) {
        // Create randomized values and save them to file
        random_init_double(matA, M * K, true);
        saveMat_double("matrixA.dat", matA, M * K);
        random_init_double(matB, K * N, true);
        saveMat_double("matrixB.dat", matB, K * N);
    } else if (newMatrix == 0) {
        // Read matrices from file
        readMat_double("matrixA.dat", matA, N*K);
        readMat_double("matrixB.dat", matB, K*M); 
    }
    
    // unsync the I/O of C and C++
    ios_base::sync_with_stdio(false);

    // --------------------------------------------------------------------------------------------------------------
    
    // start timer
    auto t1 = chrono::steady_clock::now();
    
    for (int i=0; i<iter; i++)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, matA, K, matB, N, beta, matC, N);
    
    // end timer
    auto t2 = chrono::steady_clock::now();

    // --------------------------------------------------------------------------------------------------------------
    
    
    // Calculating total time
    if (times == 1)
        cout << (double)chrono::duration_cast<chrono::microseconds>(t2 - t1).count()/1000000.0f/iter << " ";

    if (printOp == 1)
        printMat_double(matC, N, M);
    
    
    mkl_free(matA);
    mkl_free(matB);
    mkl_free(matC);
    
    return 0;   
}