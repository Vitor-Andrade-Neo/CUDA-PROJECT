#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void calcnorma(float *q_g, float *k_g, float *norma_q, float *norma_k, int n, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum_q = 0.0f, sum_k = 0.0f;
        for (int col = 0; col < m; col++) {
            sum_q += q_g[row * m + col] * q_g[row * m + col];
            sum_k += k_g[row * m + col] * k_g[row * m + col];
        }
        norma_q[row] = sqrtf(sum_q);
        norma_k[row] = sqrtf(sum_k);
    }
}

__global__ void scale(float *q_g, float *k_g, float *norma_q, float *norma_k, int n, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < m) {
        q_g[row * m + col] /= norma_q[row];
        k_g[row * m + col] /= norma_k[row];
    }
}

int main() {
    float *q_c, *k_c;
    float *q_g, *k_g;
    float *norma_q, *norma_k;
    float *norma_q_g, *norma_k_g;

    int n = 2;
    int m = 3;

    q_c = (float *)malloc(n * m * sizeof(float));
    k_c = (float *)malloc(n * m * sizeof(float));
    norma_q = (float *)malloc(n * sizeof(float));
    norma_k = (float *)malloc(n * sizeof(float));

    cudaMalloc(&q_g, n * m * sizeof(float));
    cudaMalloc(&k_g, n * m * sizeof(float));
    cudaMalloc(&norma_q_g, n * sizeof(float));
    cudaMalloc(&norma_k_g, n * sizeof(float));

    float init_q[] = {1, 2, 3, 4, 1, 0};
    float init_k[] = {1, 0, 7, 2, 0, 1};

    memcpy(q_c, init_q, n * m * sizeof(float));
    memcpy(k_c, init_k, n * m * sizeof(float));

    cudaMemcpy(q_g, q_c, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(k_g, k_c, n * m * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(8, 8);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    calcnorma<<<1, n>>>(q_g, k_g, norma_q_g, norma_k_g, n, m);
    cudaDeviceSynchronize();

    cudaMemcpy(norma_q, norma_q_g, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(norma_k, norma_k_g, n * sizeof(float), cudaMemcpyDeviceToHost);

    scale<<<gridDim, blockDim>>>(q_g, k_g, norma_q_g, norma_k_g, n, m);
    cudaDeviceSynchronize();


    cudaMemcpy(q_c, q_g, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_c, k_g, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Scaled matrix q:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%f ", q_c[i * m + j]);
            }
            printf("\n");
    }

    printf("\nScaled matrix k:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%f ", k_c[i * m + j]);
            }
            printf("\n");
    }

    cudaFree(q_g);
    cudaFree(k_g);
    cudaFree(norma_q_g);
    cudaFree(norma_k_g);
    free(q_c);
    free(k_c);
    free(norma_q);
    free(norma_k);

    return 0;
}
