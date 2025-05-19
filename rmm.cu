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

#define TILE 16

__global__ void rmm_kernel(
    const int* __restrict__ A,
    const int* __restrict__ B,
          int* __restrict__ C,
    int M, int N, int K)
{
    // global-reduced-matrix coords
    int out_row = blockIdx.y * TILE + threadIdx.y;  // 0 .. M/2-1
    int out_col = blockIdx.x * TILE + threadIdx.x;  // 0 .. K/2-1

    // mark who's actually computing a valid C element
    bool active = (out_row < M/2 && out_col < K/2);

    // precompute the two A-row bases and two B-col offsets (even if inactive)
    int a0_base = (out_row*2    ) * N;
    int a1_base = (out_row*2 + 1) * N;
    int b0_off  = out_col*2;
    int b1_off  = b0_off + 1;

    int sum = 0;

    // shared-memory tiles: 2 rows × TILE columns, and TILE rows × 2 columns
    __shared__ int sA[2*TILE][TILE];
    __shared__ int sB[TILE][2*TILE];

    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int kBase = t * TILE;

        // --- load A (two rows) ---
        int col = kBase + threadIdx.x;
        if (col < N) {
            sA[threadIdx.y*2    ][threadIdx.x] = A[a0_base + col];
            sA[threadIdx.y*2 + 1][threadIdx.x] = A[a1_base + col];
        } else {
            sA[threadIdx.y*2    ][threadIdx.x] = 0;
            sA[threadIdx.y*2 + 1][threadIdx.x] = 0;
        }

        // --- load B (two cols) ---
        int brow = kBase + threadIdx.y;
        if (brow < N) {
            sB[threadIdx.y][threadIdx.x*2    ] = B[brow*K + b0_off];
            sB[threadIdx.y][threadIdx.x*2 + 1] = B[brow*K + b1_off];
        } else {
            sB[threadIdx.y][threadIdx.x*2    ] = 0;
            sB[threadIdx.y][threadIdx.x*2 + 1] = 0;
        }

        // *all* threads sync before we read from sA/sB
        __syncthreads();

        // accumulate only if this thread is active
        if (active) {
            int limit = min(TILE, N - kBase);
            for (int k = 0; k < limit; ++k) {
                int A0 = sA[threadIdx.y*2    ][k];
                int A1 = sA[threadIdx.y*2 + 1][k];
                int B0 = sB[k][threadIdx.x*2    ];
                int B1 = sB[k][threadIdx.x*2 + 1];
                sum += A0*B0 + A0*B1 + A1*B0 + A1*B1;
            }
        }

        // *all* threads sync before we overwrite sA/sB
        __syncthreads();
    }

    // finally write C if active
    if (active) {
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
    dim3 blockDim(TILE, TILE);  // TILExTILE threads per block
    dim3 gridDim((K/2 + TILE - 1) / TILE, 
                 (M/2 + TILE - 1) / TILE);

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