/*
============================================================================
Filename    : assignment3.c
Author      : PARSA, EPFL
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;
#include "utility.h"

void rmm_cpu(int *matA, int *matB, int *matC, int M, int N, int K);
void rmm_gpu(int *matA, int *matB, int *matC, int M, int N, int K);

int main (int argc, const char *argv[]) {
    if(argc != 5) {
        printf("Usage: %s <M> <N> <K> <0|1>\n", argv[0]);
        return 1;
    }
    
    /* Step 1: Read the values of M, N and K from the command line arguments. */
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int debug = atoi(argv[4]);

    if(M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
        printf("M, N and K must be even\n");
        return 1;
    }

    /* Step 2: Generates and initializes matrices A and B with random values. */
    int *matA, *matB, *matC_cpu, *matC_gpu;
    matA = (int *) malloc(M * N * sizeof(int));
    matB = (int *) malloc(N * K * sizeof(int));
    matC_cpu = (int *) malloc((M/2) * (K/2) * sizeof(int));
    matC_gpu = (int *) malloc((M/2) * (K/2) * sizeof(int));

    init_mat(matA, M, N, 0);
    init_mat(matB, N, K, 1);        
    init_mat(matC_cpu, M/2, K/2, -1);   // -1 indicates that matrix is initialized with 0s
    init_mat(matC_gpu, M/2, K/2, -1);   // -1 indicates that matrix is initialized with 0s

    if(debug) {
        display_matrix(matA, M, N, "A");
        display_matrix(matB, N, K, "B");
    }

    /* Reset Device */
    cudaDeviceReset();

    /* Start Timer for CPU */
    set_clock();
    rmm_cpu(matA, matB, matC_cpu, M, N, K);
    double cpu_time = elapsed_time();
    cout << "CPU Total time taken: " << setprecision(4) << cpu_time << "s" << endl;
    write_csv(matC_cpu, M/2, K/2, "matC_cpu.csv");

    /* Reset Device */
    cudaDeviceReset();

    /* Start Timer for GPU */
    set_clock();
    rmm_gpu(matA, matB, matC_gpu, M, N, K);
    double gpu_time = elapsed_time();
    cout << "GPU Total time taken: " << setprecision(4) << gpu_time << "s" << endl;
    write_csv(matC_gpu, M/2, K/2, "matC.csv");

    /* Print output matrices if debug */
    if(debug) {
        display_matrix(matC_cpu, M/2, K/2, "C (CPU)");
        display_matrix(matC_gpu, M/2, K/2, "C (GPU)");
    }
    
    /* Free allocated memory */
    free(matA);
    free(matB);
    free(matC_cpu);
    free(matC_gpu);

    return 0;
}
