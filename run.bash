#!/bin/bash

# ------------------------------------------------------------------------------------------------------------
# Compilation without PAPI libraries
# ------------------------------------------------------------------------------------------------------------

g++ -O3 -m64 MatMul_serial.cpp MatMul.cpp -o MatMul_serial
g++ -O3 -m64 -fopenmp MatMul_Parallel_OMP.cpp MatMul.cpp -o MatMul_Parallel_OMP
g++ -O3 -m64 -mavx2 -mfma MatMul_Parallel_AVX.cpp MatMul.cpp -o MatMul_Parallel_AVX 
g++ -O3 -m64 -mavx2 -mfma -fopenmp MatMul_Parallel_AVX_omp.cpp MatMul.cpp -o MatMul_Parallel_AVX_omp
g++ -m64 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma MatMul_Parallel_AVX512.cpp MatMul.cpp -o MatMul_Parallel_AVX512
g++ -m64 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -fopenmp MatMul_Parallel_AVX512_omp.cpp MatMul.cpp -o MatMul_Parallel_AVX512_omp
nvcc MatMul_Parallel_cuda.cu MatMul.cpp -o MatMul_Parallel_cuda
nvcc -lcublas -o MatMul_Parallel_cuBLAS MatMul_Parallel_cuBLAS.cpp MatMul.cpp

# ------------------------------------------------------------------------------------------------------------
#sh -c "echo 2 > /proc/sys/kernel/perf_event_paranoid"
# ------------------------------------------------------------------------------------------------------------

declare -a sizes=(64 128 256 512 1024)

# ------------------------------------------------------------------------------------------------------------
# Times
# ------------------------------------------------------------------------------------------------------------
echo "" > times.txt
for i in "${!sizes[@]}"
do
    ./MatMul_serial ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 1 >> times.txt
    ./MatMul_Parallel_OMP ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 1 >> times.txt
    ./MatMul_Parallel_AVX ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 1 >> times.txt
    ./MatMul_Parallel_AVX_omp ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 1 >> times.txt
    ./MatMul_Parallel_AVX512 ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 1 >> times.txt
    ./MatMul_Parallel_AVX512_omp ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 1 >> times.txt
    ./MatMul_Parallel_cuda ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 1 >> times.txt
    ./MatMul_Parallel_cuBLAS ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 1 >> times.txt
    echo -e "" >> times.txt
done

# ------------------------------------------------------------------------------------------------------------
# Perf Power Consumption
# ------------------------------------------------------------------------------------------------------------
for i in "${!sizes[@]}"
do
    perf stat -o power_serial_${sizes[$i]}.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./MatMul_serial ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0 
    perf stat -o power_parallel_omp_${sizes[$i]}.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./MatMul_Parallel_OMP ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0 
    perf stat -o power_parallel_avx_${sizes[$i]}.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./MatMul_Parallel_AVX ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0 
    perf stat -o power_parallel_avx_omp_${sizes[$i]}.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./MatMul_Parallel_AVX_omp ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0 
    perf stat -o power_parallel_avx512_${sizes[$i]}.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./MatMul_Parallel_AVX512 ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0 
    perf stat -o power_parallel_avx512_omp_${sizes[$i]}.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./MatMul_Parallel_AVX512_omp ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0 
    perf stat -o power_parallel_cuda_${sizes[$i]}.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./MatMul_Parallel_cuda ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0 
    perf stat -o power_parallel_cublas_${sizes[$i]}.txt -a -r 1 -e "power/energy-pkg/" -e "power/energy-ram/" ./MatMul_Parallel_cuBLAS ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0 
done


# ------------------------------------------------------------------------------------------------------------
# Compilation with PAPI libraries
# ------------------------------------------------------------------------------------------------------------
cd PAPI
g++ -O3 -m64 -L /opt/papi/lib -lpapi -I /opt/papi/include MatMul_serial.cpp MatMul.cpp handle_error.c -o MatMul_serial
g++ -O3 -m64 -fopenmp -L /opt/papi/lib -lpapi -I /opt/papi/include MatMul_Parallel_OMP.cpp MatMul.cpp handle_error.c -o MatMul_Parallel_OMP
g++ -O3 -m64 -mavx2 -mfma -L /opt/papi/lib -lpapi -I /opt/papi/include MatMul_Parallel_AVX.cpp MatMul.cpp handle_error.c -o MatMul_Parallel_AVX 
g++ -O3 -m64 -mavx2 -mfma -fopenmp -L /opt/papi/lib -lpapi -I /opt/papi/include MatMul_Parallel_AVX_omp.cpp MatMul.cpp handle_error.c -o MatMul_Parallel_AVX_omp
g++ -m64 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -L /opt/papi/lib -lpapi  -I /opt/papi/include MatMul_Parallel_AVX512.cpp MatMul.cpp handle_error.c -o MatMul_Parallel_AVX512
g++ -m64 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -fopenmp -L /opt/papi/lib -lpapi -I /opt/papi/include MatMul_Parallel_AVX512_omp.cpp MatMul.cpp handle_error.c -o MatMul_Parallel_AVX512_omp
nvcc -L /opt/papi/lib -lpapi -I /opt/papi/include MatMul_Parallel_cuda.cu MatMul.cpp handle_error.c -o MatMul_Parallel_cuda
nvcc -lcublas -L /opt/papi/lib -lpapi -I /opt/papi/include -o MatMul_Parallel_cuBLAS MatMul_Parallel_cuBLAS.cpp MatMul.cpp handle_error.c

# ------------------------------------------------------------------------------------------------------------
# PAPI Power Consumption
# ------------------------------------------------------------------------------------------------------------

for i in "${!sizes[@]}"
do
    export PAPI_MULTIPLEX=1
    export PAPI_EVENTS="rapl::RAPL_ENERGY_PKG,rapl::RAPL_ENERGY_DRAM"
    ./MatMul_serial ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_OMP ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_AVX ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_AVX_omp ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_AVX512 ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_AVX512_omp ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_cuda ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_cuBLAS ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
done