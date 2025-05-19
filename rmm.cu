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
    // figure out which output element this thread will write:
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..M/2-1
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..K/2-1

    // precompute the two A-row bases and two B-col offsets:
    int a_row_base0 = (out_row*2    ) * N;
    int a_row_base1 = (out_row*2 + 1) * N;
    int b_col_base  = out_col*2;

    int sum = 0;

    // how many TILES to cover the N dimension?
    int numTiles = (N + TILE - 1) / TILE;

    // shared memory for one TILE×TILE slice of A+ B
    __shared__ int sA[TILE][TILE];
    __shared__ int sB[TILE][TILE];

    for (int t = 0; t < numTiles; ++t) {
        int k_base = t * TILE;

        // load one element of A in each of the two needed rows
        int a0 = 0, a1 = 0;
        int global_k = k_base + threadIdx.x;
        if (out_row < M/2 && global_k < N) {
            a0 = A[a_row_base0 + global_k];
            a1 = A[a_row_base1 + global_k];
        }
        sA[threadIdx.y*2    ][threadIdx.x] = a0;  // fold two rows into shared
        sA[threadIdx.y*2 + 1][threadIdx.x] = a1;

        // load one element of B in each of the two needed columns
        int b0 = 0, b1 = 0;
        int global_k_y = k_base + threadIdx.y;
        if (global_k_y < N && out_col < K/2) {
            b0 = B[global_k_y * K + b_col_base    ];
            b1 = B[global_k_y * K + b_col_base + 1];
        }
        sB[threadIdx.y][threadIdx.x*2    ] = b0;  // fold two cols into shared
        sB[threadIdx.y][threadIdx.x*2 + 1] = b1;

        // *all* threads sync here:
        __syncthreads();

        // accumulate over this tile
        int limit = min(TILE, N - k_base);
        for (int k = 0; k < limit; ++k) {
            // two A-rows come from sA[k][…], two B-cols from sB[…][k]
            int a_0 = sA[k][threadIdx.x*2    ];
            int a_1 = sA[k][threadIdx.x*2 + 1];
            int b_0 = sB[threadIdx.y*2    ][k];
            int b_1 = sB[threadIdx.y*2 + 1][k];

            sum += a_0 * b_0
                 + a_0 * b_1
                 + a_1 * b_0
                 + a_1 * b_1;
        }

        // and sync before we overwrite shared memory next tile:
        __syncthreads();
    }

    // finally write out the result (guard in case grid > exact size)
    if (out_row < M/2 && out_col < K/2) {
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
    // Using 32x32 thread blocks for better shared memory utilization
    dim3 blockDim(32, 32);  // 1024 threads per block
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