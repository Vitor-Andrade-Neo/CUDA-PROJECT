#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

#define Br 64 
#define Bc 64 

__global__ void flash_attention2_fwd(float* Q, float* K, float* V, float* O, float* L, float* M, float scale, int N, int d) {
    int row = blockIdx.x * Br + threadIdx.x;
    int col = threadIdx.y;
    
    extern __shared__ float shared_mem[];
    float* Q_shared = shared_mem;
    float* K_shared = &shared_mem[Br * d];
    float* V_shared = &shared_mem[(Br + Bc) * d];
    float* O_shared = &shared_mem[(2 * Br + Bc) * d];
    float* L_shared = &shared_mem[(2 * (Br + Bc) * d)];
    float* M_shared = &shared_mem[(2 * (Br + Bc) * d) + Br];
    
    if (threadIdx.y == 0) {
        M_shared[threadIdx.x] = -INFINITY;
        L_shared[threadIdx.x] = 0.0f;
        for (int j = 0; j < d; j++) {
            O_shared[threadIdx.x * d + j] = 0.0f;
        }
    }
    
    __syncthreads();

    for (int start_kv = 0; start_kv < N; start_kv += Bc) {
        if (row < N && col < d) {
            Q_shared[threadIdx.x * d + col] = Q[row * d + col];
            K_shared[threadIdx.x * d + col] = K[(start_kv + threadIdx.x) * d + col];
            V_shared[threadIdx.x * d + col] = V[(start_kv + threadIdx.x) * d + col];
        }
        __syncthreads();
        
        float S = 0.0f;
        for (int i = 0; i < d; i++) {
            S += Q_shared[threadIdx.x * d + i] * K_shared[col * d + i];
        }
        S *= scale;
        
        float mij = fmaxf(M_shared[threadIdx.x], S);
        float P = expf(S - mij);
        L_shared[threadIdx.x] = L_shared[threadIdx.x] * expf(M_shared[threadIdx.x] - mij) + P;
        for (int i = 0; i < d; i++) {
            O_shared[threadIdx.x * d + i] = O_shared[threadIdx.x * d + i] * expf(M_shared[threadIdx.x] - mij) + P * V_shared[col * d + i];
        }
        M_shared[threadIdx.x] = mij;
        __syncthreads();
    }
    
    for (int col = 0; col < d; col++) {
        O[row * d + col] = O_shared[threadIdx.x * d + col] / L_shared[threadIdx.x];
    }
    L[row] = logf(L_shared[threadIdx.x]);
}

int main() {
    int N = 128;
    int d = 64;
    int Tr = N / Br, Tc = N / Bc;
    float *Q, *K, *V, *O, *L, *M;
    
    cudaMallocManaged(&Q, N * d * sizeof(float));
    cudaMallocManaged(&K, N * d * sizeof(float));
    cudaMallocManaged(&V, N * d * sizeof(float));
    cudaMallocManaged(&O, N * d * sizeof(float));
    cudaMallocManaged(&L, Tr * Br * sizeof(float));
    cudaMallocManaged(&M, Tr * Br * sizeof(float));
    
    for (int i = 0; i < N * d; i++) {
        Q[i] = static_cast<float>(rand()) / RAND_MAX;
        K[i] = static_cast<float>(rand()) / RAND_MAX;
        V[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    cudaMemset(L, 0, Tr * Br * sizeof(float));
    cudaMemset(O, 0, N * d * sizeof(float));
    
    dim3 block(Br, Bc);
    dim3 grid((N + Br - 1) / Br);
    size_t shared_memory_size = (2 * (Br * d) + 2 * (Bc * d) + 2 * Br) * sizeof(float);
    float scale = 1.0f / sqrtf((float)d);
    
    flash_attention2_fwd<<<grid, block, shared_memory_size>>>(Q, K, V, O, L, M, scale, N, d);
    cudaDeviceSynchronize();
    
    cout << "Resultado do Flash Attention 2:" << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            cout << O[i * d + j] << " ";
        }
        cout << endl;
    }
    
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(O);
    cudaFree(L);
    cudaFree(M);
    
    return 0;
}
