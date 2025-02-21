#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void MultiMat(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float soma = 0;
        for (int k = 0; k < n; k++) {
            soma += a[i* n + k] * b[k * n + j];
        }
        c[i * n + j] = soma;
    }
}

int main() {
    int n, x, y;
    cin >> n;

    float *a, *b, *c;
    a = (float*)malloc(n * n * sizeof(float));
    b = (float*)malloc(n * n * sizeof(float));
    c = (float*)malloc(n * n * sizeof(float));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> x >> y;
            a[i * n + j] = x;
            b[i * n + j] = y;
            c[i * n + j] = 0.0f;
        }
    }

    float *da, *db, *dc;
    cudaMalloc((void **)&da, n * n * sizeof(float));
    cudaMalloc((void **)&db, n * n * sizeof(float));
    cudaMalloc((void **)&dc, n * n * sizeof(float));

    cudaMemcpy(da, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 16);
    dim3 dimGrid((n + 31) / 32, (n + 15) / 16);
    
    MultiMat<<<dimGrid, dimBlock>>>(da, db, dc, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dc, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("C:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", c[i * n + j]);
        }
        printf("\n");
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);

    return 0;
}
