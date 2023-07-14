#!/bin/bash

root=`pwd`

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
cd $root/PAPI

g++ -O3 -m64 -I /opt/papi/include handle_error.c MatMul_serial.cpp MatMul.cpp /opt/papi/lib/libpapi.a -o MatMul_serial_papi   
g++ -O3 -m64 -fopenmp -I /opt/papi/include handle_error.c MatMul_Parallel_OMP.cpp MatMul.cpp /opt/papi/lib/libpapi.a -o MatMul_Parallel_OMP_papi
g++ -O3 -m64 -mavx2 -mfma -I /opt/papi/include handle_error.c  MatMul_Parallel_AVX.cpp MatMul.cpp /opt/papi/lib/libpapi.a -o MatMul_Parallel_AVX_papi 
g++ -O3 -m64 -mavx2 -mfma -fopenmp -I /opt/papi/include handle_error.c MatMul_Parallel_AVX_omp.cpp MatMul.cpp /opt/papi/lib/libpapi.a -o MatMul_Parallel_AVX_omp_papi
g++ -m64 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -I /opt/papi/include handle_error.c MatMul_Parallel_AVX512.cpp MatMul.cpp /opt/papi/lib/libpapi.a -o MatMul_Parallel_AVX512_papi
g++ -m64 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -fopenmp -I /opt/papi/include handle_error.c MatMul_Parallel_AVX512_omp.cpp MatMul.cpp /opt/papi/lib/libpapi.a -o MatMul_Parallel_AVX512_omp_papi
nvcc -rdc=true -I /opt/papi/include MatMul_Parallel_cuda.cu MatMul.cpp -L /opt/papi/lib -lpapi -o MatMul_Parallel_cuda_papi  
nvcc -lcublas -I /opt/papi/include handle_error.c /opt/papi/lib/libpapi.a -o MatMul_Parallel_cuBLAS_papi MatMul_Parallel_cuBLAS.cpp MatMul.cpp 

# ------------------------------------------------------------------------------------------------------------
# PAPI Power Consumption
# ------------------------------------------------------------------------------------------------------------

for i in "${!sizes[@]}"
do
    export PAPI_MULTIPLEX=1
    export PAPI_EVENTS="rapl::RAPL_ENERGY_PKG,rapl::RAPL_ENERGY_DRAM"
    ./MatMul_serial_papi  ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_OMP_papi  ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_AVX_papi  ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_AVX_omp_papi  ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_AVX512_papi  ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_AVX512_omp_papi  ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_cuda_papi  ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
    ./MatMul_Parallel_cuBLAS_papi  ${sizes[$i]} ${sizes[$i]} ${sizes[$i]} 1 1 0 0
done