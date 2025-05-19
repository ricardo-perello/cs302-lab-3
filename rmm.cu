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

/* CUDA Kernel for RMM with shared memory */
__global__ void rmm_kernel(int *matA, int *matB, int *matC, int M, int N, int K)
{
    // Shared memory for tile of matrix A and B
    __shared__ int s_matA[32][32];  // 32x32 tile of A
    __shared__ int s_matB[32][32];  // 32x32 tile of B
    
    // Calculate global indices
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int jdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Local indices within shared memory
    int local_y = threadIdx.y;
    int local_x = threadIdx.x;
    
    // Accumulator for this thread's result
    int sum = 0;
    
    // Number of tiles needed
    int numTiles = (N + 31) / 32;
    
    // Check if within bounds
    if (idx < M/2 && jdx < K/2) {
        // Loop over tiles
        for (int t = 0; t < numTiles; t++) {
            // Load tile of A into shared memory
            if (idx*2 + local_y < M && t*32 + local_x < N) {
                s_matA[local_y][local_x] = matA[(idx*2 + local_y)*N + t*32 + local_x];
            } else {
                s_matA[local_y][local_x] = 0;
            }
            
            // Load tile of B into shared memory
            if (t*32 + local_y < N && jdx*2 + local_x < K) {
                s_matB[local_y][local_x] = matB[(t*32 + local_y)*K + jdx*2 + local_x];
            } else {
                s_matB[local_y][local_x] = 0;
            }
            
            // Synchronize to ensure all threads have loaded their data
            __syncthreads();
            
            // Compute partial sum for this tile
            for (int k = 0; k < 32; k++) {
                if (t*32 + k < N) {
                    for (int aoff = 0; aoff < 2; aoff++) {
                        for (int boff = 0; boff < 2; boff++) {
                            sum += s_matA[aoff][k] * s_matB[k][boff];
                        }
                    }
                }
            }
            
            // Synchronize before loading next tile
            __syncthreads();
        }
        
        // Write result
        matC[idx*(K/2) + jdx] = sum;
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