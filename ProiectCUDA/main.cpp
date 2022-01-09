// CudaTestRun.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdlib.h>
#include <time.h> 
#include <fstream>
#include <queue>
#include "./Header.cuh"

using namespace std;
using std::queue;


void printMatrix(double** V, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            cout << V[i][j] << " ";
        cout << endl;
    }
}

int matriceEgale(double** A, double** B, int M, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (A[i][j] != B[i][j])
                return 0;
    return 1;
}

void genereazaSiIncarcaMatrice(int M, int N) {
    srand(time(NULL));
    ofstream f("data.txt");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            f << rand() % 10 << " ";
        }
        f << endl;
    }
    f.close();
}

double** citesteMatrice(const int M, const int N) {
    double** F = new double* [M];
    for (int i = 0; i < M; ++i)
        F[i] = new double[N];
    ifstream f("data.txt");
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            f >> F[i][j];
    return F;
}

void transformaLinie(double** F, double** W, double** V, int M, int N, int m, int n, int i) {
    int k = m / 2, l = n / 2, ap, am, bp, bm;
    double v = 0;
    for (int j = 0; j < N; j++) {
        for (int a = 0; a <= k; a++) {
            ap = i + a;
            am = i - a;
            if (am < 0)
                am = 0;
            if (ap >= M)
                ap = M - 1;
            for (int b = 0; b <= l; b++) {
                if (a == 0 && b == 0)
                    v = F[i][j] * W[k][l];
                else {
                    bp = j + b;
                    bm = j - b;
                    if (bm < 0)
                        bm = 0;
                    if (bp >= N)
                        bp = N - 1;
                    if (a == 0) {
                        v += F[i][bp] * W[k][l + b];
                        v += F[i][bm] * W[k][l - b];
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
        V[i][j] = v;
    }
}

void functieRun(double** F, double** W, double** V, int start, int M, int N, int m, int n, int p) {
    for (int i = start; i < M; i += p) {
        transformaLinie(F, W, V, M, N, m, n, i);
    }
}

void transformaSecvential(double** F, double** W, double** V, int M, int N, int m, int n) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            transformaLinie(F, W, V, M, N, m, n, i);
}

int main()
{
    int M, N, m, n, p;
    M = 10;
    N = 10;
    m = 3;
    n = 3;
    p = 4;
    double** F;
    genereazaSiIncarcaMatrice(M,N);
    F = citesteMatrice(M, N);
    double** V = new double* [M];
    double** V1 = new double* [M];
    for (int i = 0; i < M; ++i) {
        V[i] = new double[N];
        V1[i] = new double[N];
    }
    double** W = new double* [m];
    for (int i = 0; i < m; ++i) {
        W[i] = new double[n];
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            W[i][j] = (double)1 / (m * n);

    transformaSecvential(F, W, V, M, N, m, n);

    printMatrix(V, M, N);
    cout << endl << endl;

    kernel(F, W, V1, M, N, m, n);
    printMatrix(V1, M, N);

    cout<<"Egalitatea matricelor: "<< matriceEgale(V, V1, M, N)<<endl;

    for (int i = 0; i < M; i++) {
        delete[] F[i];
        delete[] V[i];
        delete[] V1[i];
    }
    delete[] F;
    delete[] V;
    delete[] V1;
    for (int i = 0; i < m; i++)
        delete[] W[i];
    delete[] W;

    return 0;
}

