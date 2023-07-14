#pragma once

extern void random_init(float *data, int length, bool flag);
extern void printMat(const float *C, int N, int M);
extern void readMat(const char path[100], float *vector, int length);
extern void saveMat(const char path[100], float *vector, int length);
extern void matInv(float *mat, int rows, int cols);

extern void random_init_double(double *data, int length, bool flag);
extern void saveMat_double(const char path[100], double *vector, int length);
extern void readMat_double(const char path[100], double *vector, int length);
extern void printMat_double(const double *C, int N, int M);

extern bool verification(const float *C_serial, const float *C_parallel, size_t length);