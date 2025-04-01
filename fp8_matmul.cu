#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <stdint.h>

#define M 4
#define K 4
#define N 4

// -------- FP8 Quantização simples (simulada) --------

// Simula E4M3 quantizando com escala
__device__ __host__ uint8_t quantize_fp8(float val, float scale) {
    int q = roundf(val * scale);
    q = max(-8, min(7, q)); // E4M3 vai de -8 a +7 em inteiros (simplificado)
    return static_cast<uint8_t>(q & 0xFF); // armazenado como uint8_t
}

// Simula desquantização FP8 → FP32
__device__ __host__ float dequantize_fp8(uint8_t val, float scale) {
    int8_t sval = static_cast<int8_t>(val);
    return static_cast<float>(sval) / scale;
}

// -------- CUDA Kernel de MatMul (com FP8 "simulada") --------

__global__ void matmul_fp8_kernel(
    const uint8_t* A, const uint8_t* B, float* C,
    float scaleA, float scaleB, int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = dequantize_fp8(A[row * K + k], scaleA);
            float b = dequantize_fp8(B[k * N + col], scaleB);
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

// -------- Host Code --------

int main() {
    float h_A[M * K], h_B[K * N];
    uint8_t h_Aq[M * K], h_Bq[K * N];
    float *d_C, h_C[M * N];
    uint8_t *d_A, *d_B;

    // Inicializa entradas com valores pequenos
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand() % 10) / 10.0f;

    // Escalas manuais
    float scaleA = 16.0f; // simples inverso do range
    float scaleB = 16.0f;

    // Quantiza A e B para FP8 simulado
    for (int i = 0; i < M * K; ++i) h_Aq[i] = quantize_fp8(h_A[i], scaleA);
    for (int i = 0; i < K * N; ++i) h_Bq[i] = quantize_fp8(h_B[i], scaleB);

    // Aloca e copia para GPU
    cudaMalloc(&d_A, M * K * sizeof(uint8_t));
    cudaMalloc(&d_B, K * N * sizeof(uint8_t));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_Aq, M * K * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_Bq, K * N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_fp8_kernel<<<grid, block>>>(d_A, d_B, d_C, scaleA, scaleB, M, K, N);

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print resultado
    std::cout << "Resultado da MatMul (simulada FP8):\n";
    for (int i = 0; i < M * N; ++i) {
        std::cout << h_C[i] << " ";
        if ((i + 1) % N == 0) std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
