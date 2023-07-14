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
#include <immintrin.h>
#include <chrono>
#include "MatMul.h"

using namespace std;

// use these arrays to perform masked load and store
// operations for any residual columns.
const uint32_t ZR = 0;
const uint32_t MV = 0x80000000;

alignas(32) const uint32_t c_Mask0[8] {ZR, ZR, ZR, ZR, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask1[8] {MV, ZR, ZR, ZR, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask2[8] {MV, MV, ZR, ZR, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask3[8] {MV, MV, MV, ZR, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask4[8] {MV, MV, MV, MV, ZR, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask5[8] {MV, MV, MV, MV, MV, ZR, ZR, ZR};
alignas(32) const uint32_t c_Mask6[8] {MV, MV, MV, MV, MV, MV, ZR, ZR};
alignas(32) const uint32_t c_Mask7[8] {MV, MV, MV, MV, MV, MV, MV, ZR};

const uint32_t *c_MaskMovLUT[8] {
    c_Mask0, c_Mask1, c_Mask2, c_Mask3, c_Mask4, c_Mask5, c_Mask7, c_Mask7
};

void matMul_avx2(const float *A, const float *B, float *C, int N, int M, int K);

int main(int argc, char **argv) {
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

    // --------------------------------------------------------------------------------------------------------------
    
    // start timer
    auto t1 = chrono::steady_clock::now();

    for (int i=0; i<iter; i++)
        matMul_avx2(matA, matB, matC, N, M, K);
    
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

void matMul_avx2(const float *A, const float *B, float *C, int N, int M, int K) {
    /*
        A -> N * K
        B -> K * M
        C -> N * M
        N: rows in A -> C
        M: cols in B -> C
        K: cols in A and rows in B
    */
    const size_t num_simd_elements = 8;
    size_t num_residuals_cols = N % num_simd_elements;

    // The reason this error is because res_mask has 4 elements and this must be 8 elementso
    __m256i res_mask = _mm256_load_si256( (__m256i*)c_MaskMovLUT[num_residuals_cols]);
    // Repeat for each row in C
    for (size_t i=0; i<N; i++) {
        size_t j = 0;
        // Repeat while 8 or more columns remain
        while (j + num_simd_elements <= M) {
            __m256 c_vals = _mm256_setzero_ps();
            // Calculate prodcuts for C[i][j:j+7]
            for (size_t k=0; k<K; k++) {
                __m256 a_vals = _mm256_broadcast_ss(&A[i * K + k]);
                __m256 b_vals = _mm256_loadu_ps(&B[k * M + j]);
                c_vals = _mm256_fmadd_ps(a_vals, b_vals, c_vals);
            }
            _mm256_storeu_ps(&C[i * M + j], c_vals);
            j += num_simd_elements;
        }

        if (num_residuals_cols) {
            __m256 c_vals = _mm256_setzero_ps();
            for (size_t k=0; k<K; k++) {
                __m256 a_vals = _mm256_broadcast_ss(&A[i * K + k]);
                __m256 b_vals = _mm256_maskload_ps(&B[k * M + j], res_mask);
                c_vals = _mm256_fmadd_ps(a_vals, b_vals, c_vals);
            }
            _mm256_maskstore_ps(&C[i * M + j], res_mask, c_vals);
        }
    }
}