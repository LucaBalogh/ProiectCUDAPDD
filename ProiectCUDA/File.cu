#include <iostream>
#include "cuda_runtime.h"
#include "./Header.cuh"
#include "device_launch_parameters.h"
using namespace std;

__global__ void transformareMatriceKernel(double* F, double* W, double* V, int M, int N, int m, int n) { 
    
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int col = index % N;
    int row = index / N;

    if (col < N && row < M) {
        int k = m / 2, l = n / 2, ap, am, bp, bm;
        double v = 0;
        for (int a = 0; a <= k; a++) {
            ap = row + a;
            am = row - a;
            if (am < 0)
                am = 0;
            if (ap >= M)
                ap = M - 1;
            for (int b = 0; b <= l; b++) {
                    if (a == 0 && b == 0)
                        v = F[index] * W[k*n+l];
                    else {
                        bp = col + b;
                        bm = col - b;
                        if (bm < 0)
                            bm = 0;
                        if (bp >= N)
                            bp = N - 1;
                        if (a == 0) {
                            v += F[row*N+bp] * W[k*n+(l + b)];
                            v += F[row*N+bm] * W[k*n+(l - b)];
                        }
                        else if (b == 0) {
                            v += F[ap*N+col] * W[(k + a)*n+l];
                            v += F[am*N+col] * W[(k - a)*n+l];
                        }
                        else {
                            v += F[ap*N+bp] * W[(k + a)*n+(l + b)];
                            v += F[ap*N+bm] * W[(k + a)*n+(l - b)];
                            v += F[am*N+bp] * W[(k - a)*n+(l + b)];
                            v += F[am*N+bm] * W[(k - a)*n+(l - b)];
                        }
                    }
            }    
        } 
        V[index] = v ;
    }
 
}

void kernel(double** F, double** W, double** V, int M, int N, int m, int n) {

    double* d_F, * d_W, * d_V;

    cudaMalloc((void**)&d_F, M * N * sizeof(double));
    cudaMalloc((void**)&d_V, M * N * sizeof(double));
    cudaMalloc((void**)&d_W, m * n * sizeof(double));

    double* h_F = new double[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_F[j + i * N] = F[i][j];
        }
    }

    double* h_W = new double[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            h_W[j + i * n] = W[i][j];
        }
    }

    cudaMemcpy(d_F, h_F, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, m * n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(512,1,1);
    dim3 gridSize(512 / M*N + 1, 1);

    transformareMatriceKernel << < gridSize, blockSize >> > (d_F, d_W, d_V, M, N, m, n);

    double* h_V = new double[N * M];                   
    cudaMemcpy(h_V, d_V, N * M * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
           V[i][j] = h_V[j + i * N];
        }
    }
    

    cudaFree(d_F);
    cudaFree(d_W);
    cudaFree(d_V);

    delete[] h_F, h_V, h_W;
}