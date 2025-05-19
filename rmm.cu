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
            // Unroll the inner loop by a factor of 4
            #pragma unroll 4
            for (int k = 0; k < limit; k += 4) {
                // Process 4 elements at a time
                if (k + 0 < limit) {
                    int A0_0 = sA[threadIdx.y*2    ][k + 0];
                    int A1_0 = sA[threadIdx.y*2 + 1][k + 0];
                    int B0_0 = sB[k + 0][threadIdx.x*2    ];
                    int B1_0 = sB[k + 0][threadIdx.x*2 + 1];
                    sum += A0_0*B0_0 + A0_0*B1_0 + A1_0*B0_0 + A1_0*B1_0;
                }
                if (k + 1 < limit) {
                    int A0_1 = sA[threadIdx.y*2    ][k + 1];
                    int A1_1 = sA[threadIdx.y*2 + 1][k + 1];
                    int B0_1 = sB[k + 1][threadIdx.x*2    ];
                    int B1_1 = sB[k + 1][threadIdx.x*2 + 1];
                    sum += A0_1*B0_1 + A0_1*B1_1 + A1_1*B0_1 + A1_1*B1_1;
                }
                if (k + 2 < limit) {
                    int A0_2 = sA[threadIdx.y*2    ][k + 2];
                    int A1_2 = sA[threadIdx.y*2 + 1][k + 2];
                    int B0_2 = sB[k + 2][threadIdx.x*2    ];
                    int B1_2 = sB[k + 2][threadIdx.x*2 + 1];
                    sum += A0_2*B0_2 + A0_2*B1_2 + A1_2*B0_2 + A1_2*B1_2;
                }
                if (k + 3 < limit) {
                    int A0_3 = sA[threadIdx.y*2    ][k + 3];
                    int A1_3 = sA[threadIdx.y*2 + 1][k + 3];
                    int B0_3 = sB[k + 3][threadIdx.x*2    ];
                    int B1_3 = sB[k + 3][threadIdx.x*2 + 1];
                    sum += A0_3*B0_3 + A0_3*B1_3 + A1_3*B0_3 + A1_3*B1_3;
                }
            }
        }

        // *all* threads sync before we overwrite sA/sB
        __syncthreads();
    }

    // finally write C if active using atomic add
    if (active) {
        atomicAdd(&C[out_row * (K/2) + out_col], sum);
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

    // Create two CUDA streams for overlapping operations
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // Allocate pinned host memory
    int *h_matA, *h_matB, *h_matC;
    cudaMallocHost(&h_matA, M * N * sizeof(int));
    cudaMallocHost(&h_matB, N * K * sizeof(int));
    cudaMallocHost(&h_matC, (M/2) * (K/2) * sizeof(int));

    // Copy input data to pinned memory
    memcpy(h_matA, matA, M * N * sizeof(int));
    memcpy(h_matB, matB, N * K * sizeof(int));

    // Allocate device memory
    int *d_matA, *d_matB, *d_matC;
    cudaMalloc(&d_matA, M * N * sizeof(int));
    cudaMalloc(&d_matB, N * K * sizeof(int));
    cudaMalloc(&d_matC, (M/2) * (K/2) * sizeof(int));

    // Initialize output matrix to zero
    cudaMemset(d_matC, 0, (M/2) * (K/2) * sizeof(int));

    // Calculate grid and block dimensions
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((K/2 + TILE - 1) / TILE, 
                 (M/2 + TILE - 1) / TILE);

    // Split the computation into two halves along the N dimension
    int N_half = N / 2;
    int N_remainder = N % 2;

    cudaEventRecord(cpy_H2D_start);
    
    // Stream 1: First half of the computation
    cudaMemcpyAsync(d_matA, h_matA, M * N_half * sizeof(int), cudaMemcpyHostToDevice, s1);
    cudaMemcpyAsync(d_matB, h_matB, N_half * K * sizeof(int), cudaMemcpyHostToDevice, s1);
    
    // Stream 2: Second half of the computation
    cudaMemcpyAsync(d_matA + M * N_half, h_matA + M * N_half, 
                    M * (N_half + N_remainder) * sizeof(int), cudaMemcpyHostToDevice, s2);
    cudaMemcpyAsync(d_matB + N_half * K, h_matB + N_half * K, 
                    (N_half + N_remainder) * K * sizeof(int), cudaMemcpyHostToDevice, s2);

    cudaEventRecord(comp_start);

    // Launch kernels on both streams
    rmm_kernel<<<gridDim, blockDim, 0, s1>>>(d_matA, d_matB, d_matC, M, N_half, K);
    rmm_kernel<<<gridDim, blockDim, 0, s2>>>(d_matA + M * N_half, d_matB + N_half * K, 
                                            d_matC, M, N_half + N_remainder, K);

    cudaEventRecord(comp_end);

    // Copy results back to host
    cudaEventRecord(cpy_D2H_start);
    cudaMemcpyAsync(h_matC, d_matC, (M/2) * (K/2) * sizeof(int), cudaMemcpyDeviceToHost, s1);
    cudaEventRecord(cpy_D2H_end);

    // Synchronize all operations
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);

    // Copy final result back to original memory
    memcpy(matC, h_matC, (M/2) * (K/2) * sizeof(int));

    // Free device memory
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

    // Free pinned host memory
    cudaFreeHost(h_matA);
    cudaFreeHost(h_matB);
    cudaFreeHost(h_matC);

    // Destroy streams
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    /* Display timing statistics */
    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout << "Host to Device MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout << "RMM operation takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout << "Device to Host MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;
}