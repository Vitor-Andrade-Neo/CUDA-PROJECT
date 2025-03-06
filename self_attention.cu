#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;

#define bl 16

__global__ void tp(float* A, float* B, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        B[x * rows + y] = A[y * cols + x];
    }
}

__global__ void mult(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void softmax(float* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows) {
        float maxVal = -1e20, sum = 0.0;

        for (int j = 0; j < cols; j++) 
            maxVal = fmaxf(maxVal, matrix[row * cols + j]);

        for (int j = 0; j < cols; j++) {
            matrix[row * cols + j] = expf(matrix[row * cols + j] - maxVal);
            sum += matrix[row * cols + j];
        }

        for (int j = 0; j < cols; j++) 
            matrix[row * cols + j] /= sum;
    }
}

void self(float *q, float *k, float *v, float *o, int dim, int seq){
    int size = seq * dim * sizeof(float);
    float *dq, *dk, *dv, *a, *od, *kt;

    cudaMalloc(&dq, size);
    cudaMalloc(&dk, size);
    cudaMalloc(&kt, size);
    cudaMalloc(&dv, size);
    cudaMalloc(&od, size);
    cudaMalloc(&a, seq * seq * sizeof(float));

    cudaMemcpy(dq, q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, size, cudaMemcpyHostToDevice);

    dim3 tpb(bl, bl);
    dim3 numb((seq + bl - 1) / bl, (seq + bl - 1) / bl);

    tp<<<numb, tpb>>>(dk, kt, seq, dim);
    cudaDeviceSynchronize();

    mult<<<numb, tpb>>>(dq, kt, a, seq, seq, dim);
    cudaDeviceSynchronize();

    dim3 tpbs(bl);
    dim3 numbs((seq + bl - 1) / bl);

    softmax<<<numbs, tpbs>>>(a, seq, seq);
    cudaDeviceSynchronize();

    mult<<<numb, tpb>>>(a, dv, od, seq, dim, seq);
    cudaDeviceSynchronize();

    cudaMemcpy(o, od, size, cudaMemcpyDeviceToHost);

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(a);
    cudaFree(od);
    cudaFree(kt);
}

void random(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(){
    int dim, seq;
    cin >> dim >> seq;
    int size = dim * seq;
    float *q = new float[size];
    float *k = new float[size];
    float *v = new float[size];
    float *o = new float[size];

    random(q, size); 
    random(k, size);
    random(v, size);

    self(q, k, v, o, dim, seq);

    cout << "Resultado da atenção:\n";
    for(int i=0; i<size; i++){
        cout << o[i] << " ";
        if((i+1)%seq == 0) cout << "\n";
    }

    delete[] q;
    delete[] k;
    delete[] v;
    delete[] o;

    return 0;
}
