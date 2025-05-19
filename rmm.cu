/*
============================================================================
Filename    : rmm.cu
Author      : Your name goes here
SCIPER      : Your SCIPER number
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

/* CPU Baseline */
void rmm_cpu(int *matA, int *matB, int *matC, int M, int N, int K)
{
    for(int idx = 0; idx < M/2; idx++) {
        for(int jdx = 0; jdx < K/2; jdx++) {
            matC[idx*(K/2) + jdx] = 0;
            for(int aoff = 0; aoff < 2; aoff++) {
                for(int boff = 0; boff < 2; boff++) {
                    for(int kdx = 0; kdx < N; kdx++) {
                        matC[idx*(K/2) + jdx] += matA[(idx*2 + aoff)*N + kdx] * matB[kdx*K + jdx*2 + boff];
                    }
                }
            }
        }
    }
}

#define TILE 32

__global__ void rmm_kernel(const int * __restrict__ A,
                          const int * __restrict__ B,
                                int * __restrict__ C,
                          int M, int N, int K)
{
    // Which output element (row,col) this thread computes:
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..M/2-1
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..K/2-1

    if (out_row < M/2 && out_col < K/2) {
        int sum = 0;

        // Base pointers into the two source rows of A and two cols of B
        int a0_base = (out_row*2    ) * N;
        int a1_base = (out_row*2 + 1) * N;
        int b0_off  = out_col*2;
        int b1_off  = out_col*2 + 1;

        // Accumulate all four products per k
        for (int k = 0; k < N; ++k) {
            int A0 = A[a0_base + k];
            int A1 = A[a1_base + k];
            int B0 = B[k*K + b0_off];
            int B1 = B[k*K + b1_off];
            sum += A0*B0 + A0*B1 + A1*B0 + A1*B1;
        }

        C[out_row * (K/2) + out_col] = sum;
    }
}

/* GPU Optimized Function */
void rmm_gpu(int *matA, int *matB, int *matC, int M, int N, int K)
{
    /* Cuda events for calculating elapsed time */
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);

    // Allocate device memory
    int *d_matA, *d_matB, *d_matC;
    cudaMalloc(&d_matA, M * N * sizeof(int));
    cudaMalloc(&d_matB, N * K * sizeof(int));
    cudaMalloc(&d_matC, (M/2) * (K/2) * sizeof(int));

    cudaEventRecord(cpy_H2D_start);
    // Copy input matrices to device
    cudaMemcpy(d_matA, matA, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, N * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    // Calculate grid and block dimensions
    // Using 8x8 thread blocks for debugging
    dim3 blockDim(8, 8);  // 64 threads per block
    dim3 gridDim((K/2 + blockDim.x - 1) / blockDim.x, 
                 (M/2 + blockDim.y - 1) / blockDim.y);

    cudaEventRecord(comp_start);
    // Launch kernel
    rmm_kernel<<<gridDim, blockDim>>>(d_matA, d_matB, d_matC, M, N, K);
    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    // Copy result back to host
    cudaMemcpy(matC, d_matC, (M/2) * (K/2) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    // Free device memory
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

    /* Display timing statistics */
    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout << "Host to Device MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout << "RMM operation takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout << "Device to Host MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;
}