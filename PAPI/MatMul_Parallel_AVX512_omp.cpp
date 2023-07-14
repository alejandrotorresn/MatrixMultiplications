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
#include <immintrin.h>
#include <papi.h>
#include "MatMul.h"

using namespace std;

void matMul_avx512_omp(const float *A, const float *B, float *C, int N, int M, int K);
void handle_error(int retval);

int main (int argc, char **argv) {
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
    float *matA = static_cast<float *>(_mm_malloc(sizeof(float) * M * K, 32));
    float *matB = static_cast<float *>(_mm_malloc(sizeof(float) * K * N, 32));
    float *matC = static_cast<float *>(_mm_malloc(sizeof(float) * M * N, 32));

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

    for (int i=0; i<iter; i++)
        matMul_avx512_omp(matA, matB, matC, N, M, K);

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


void matMul_avx512_omp(const float *A, const float *B, float *C, int N, int M, int K) {
    /*
        A -> N * K
        B -> K * M
        C -> N * M
        N: rows in A -> C
        M: cols in B -> C
        K: cols in A and rows in B
    */

   const size_t num_simd_elements = 16;
   size_t num_residual_cols = M % num_simd_elements;

   __mmask16 res_mask = (__mmask16)((1 << num_residual_cols) - 1);

    size_t i, j, k;
    #pragma omp parallel for private(i, k)
   for (size_t i=0; i<N; i++) {
    size_t j = 0;
    while (j + num_simd_elements <= M) {
        __m512 c_vals = _mm512_setzero_ps();
        for (size_t k=0; k<K; k++) {
            __m512 a_vals = _mm512_set1_ps(A[i * K + k]);
            __m512 b_vals = _mm512_loadu_ps(&B[k * M + j]);
            c_vals = _mm512_fmadd_ps(a_vals, b_vals, c_vals);
        }
        _mm512_storeu_ps(&C[i * M + j], c_vals);
        j += num_simd_elements;
    }

    if (num_residual_cols != 0) {
        __m512 c_vals = _mm512_setzero_ps();
        for (size_t k = 0; k<K; k++) {
            __m512 a_vals = _mm512_set1_ps(A[i * K + k]);
            __m512 b_vals = _mm512_maskz_loadu_ps(res_mask, &B[k * K + j]);
            c_vals = _mm512_fmadd_ps(a_vals, b_vals, c_vals);
        }
        _mm512_mask_storeu_ps(&C[i * M + j], res_mask, c_vals);
    }
   }
}