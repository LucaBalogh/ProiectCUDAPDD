#include "cuda_runtime.h"
#include "./Header.cuh"
#include "device_launch_parameters.h"
using namespace std;

__global__ void transformareMatriceKernel(double** F, double** W, double** V, int M, int N, int m, int n) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is within array bounds.
    if (threadID < M) {
        //transformaLinie(F, W, V, M, N, m, n, threadID);
        int k = m / 2, l = n / 2, ap, am, bp, bm;
        double v = 0;
        for (int j = 0; j < N; j++) {
            for (int a = 0; a <= k; a++) {
                ap = threadID + a;
                am = threadID - a;
                if (am < 0)
                    am = 0;
                if (ap >= M)
                    ap = M - 1;
                for (int b = 0; b <= l; b++) {
                    if (a == 0 && b == 0)
                        v = F[threadID][j] * W[k][l];
                    else {
                        bp = j + b;
                        bm = j - b;
                        if (bm < 0)
                            bm = 0;
                        if (bp >= N)
                            bp = N - 1;
                        if (a == 0) {
                            v += F[threadID][bp] * W[k][l + b];
                            v += F[threadID][bm] * W[k][l - b];
                        }
                        else if (b == 0) {
                            v += F[ap][j] * W[k + a][l];
                            v += F[am][j] * W[k - a][l];
                        }
                        else {
                            v += F[ap][bp] * W[k + a][l + b];
                            v += F[ap][bm] * W[k + a][l - b];
                            v += F[am][bp] * W[k - a][l + b];
                            v += F[am][bm] * W[k - a][l - b];
                        }
                    }
                }
            }
            V[threadID][j] = v;
        }
    }
}

void kernel(double** F, double** W, double** V, int M, int N, int m, int n) {

    // Initialize device pointers.
    double** d_F, ** d_W, ** d_V;

    // Allocate device memory.
    cudaMalloc((void***)&d_F, M * sizeof(double*));
    cudaMalloc((void***)&d_V, M * sizeof(double*)); 
    cudaMalloc((void***)&d_W, m * sizeof(double*));

    double** d_F1 = new double* [M];
    double** d_V1 = new double* [M];
    double** d_W1 = new double* [m];

    for (int i = 0; i < M; i++) {
        cudaMalloc((void**)&(d_F1[i]), N * sizeof(double));
        cudaMalloc((void**)&(d_V1[i]), N * sizeof(double));
        cudaMemcpy(d_F1[i], F[i], N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V1[i], V[i], N * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_F, d_F1, M * sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, d_V1, M * sizeof(double*), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < m; i++) {
        cudaMalloc((void**)&(d_W1[i]), n * sizeof(double));
        cudaMemcpy(d_W1[i], W[i], n * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_W, d_W1, m * sizeof(double*), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / M, 1);

    // Launch CUDA kernel.
    transformareMatriceKernel << < gridSize, blockSize >> > (d_F,d_W,d_V,M,N,m,n);

    // Copy result array c back to host memory.  
    cudaMemcpy(d_V1, d_V, M * sizeof(double*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        cudaMemcpy(V[i], d_V1[i], N * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(V, d_V, M * sizeof(double*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        cudaFree(d_F1[i]);
        cudaFree(d_V1[i]);
    }
    cudaFree(d_F);
    cudaFree(d_V);
    for (int i = 0; i < m; i++)
        cudaFree(d_W1[i]);
    cudaFree(d_W);

    delete[] d_F1;
    delete[] d_V1;
    delete[] d_W1;
}