#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;
#define TILE_SIZE 64

__global__ void flash_attention_fwd(float* q, float* k, float* v, float* o, float* m, float soft, int seqlen, int ddim) {
    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row >= seqlen) return;
    
    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem;
    float* k_shared = q_shared + TILE_SIZE * ddim;
    float* v_shared = k_shared + TILE_SIZE * ddim;
    float* o_shared = v_shared + TILE_SIZE * ddim;
    
    float mi = -INFINITY;
    float li = 1.0f;
    
    for (int start_kv = 0; start_kv < seqlen; start_kv += TILE_SIZE) {
        int col = threadIdx.y;
        if (col < ddim) {
            q_shared[threadIdx.x * ddim + col] = q[row * ddim + col];
            k_shared[threadIdx.x * ddim + col] = k[(start_kv + threadIdx.x) * ddim + col];
            v_shared[threadIdx.x * ddim + col] = v[(start_kv + threadIdx.x) * ddim + col];
        }
        __syncthreads();
        
        float qk = 0.0f;
        for (int i = 0; i < ddim; i++) {
            qk += q_shared[threadIdx.x * ddim + i] * k_shared[col * ddim + i];
        }
        
        float mij = fmaxf(mi, qk);
        qk = qk * soft - mij;
        float p = expf(qk);
        li = li * expf(mi - mij) + p;
        o_shared[threadIdx.x * ddim + col] = o_shared[threadIdx.x * ddim + col] * expf(mi - mij) + p * v_shared[col * ddim + threadIdx.y];
        mi = mij;
        __syncthreads();
    }
    
    for (int col = 0; col < ddim; col++) {
        o[row * ddim + col] = o_shared[threadIdx.x * ddim + col] / li;
    }
}

void run_attention(int seqlen, int ddim) {
    float *q, *k, *v, *o, *m;
    cudaMallocManaged(&q, seqlen * ddim * sizeof(float));
    cudaMallocManaged(&k, seqlen * ddim * sizeof(float));
    cudaMallocManaged(&v, seqlen * ddim * sizeof(float));
    cudaMallocManaged(&o, seqlen * ddim * sizeof(float));
    cudaMallocManaged(&m, seqlen * sizeof(float));
    
    for (int i = 0; i < seqlen * ddim; i++) {
        q[i] = static_cast<float>(rand()) / RAND_MAX;
        k[i] = static_cast<float>(rand()) / RAND_MAX;
        v[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((seqlen + TILE_SIZE - 1) / TILE_SIZE);
    size_t shared_memory_size = 4 * TILE_SIZE * ddim * sizeof(float);
    
    float soft = 1.0f / sqrtf((float)ddim);
    flash_attention_fwd<<<grid, block, shared_memory_size>>>(q, k, v, o, m, soft, seqlen, ddim);
    cudaDeviceSynchronize();
    
    std::cout << "Resultado do Flash Attention:" << std::endl;
    for (int i = 0; i < seqlen; i++) {
        for (int j = 0; j < ddim; j++) {
            std::cout << o[i * ddim + j] << " ";
        }
        std::cout << std::endl;
    }
    
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(o);
    cudaFree(m);
}

int main() {
    int seqlen = 128;
    int ddim = 64;
    run_attention(seqlen, ddim);
    std::cout << "Flash Attention CUDA executado com sucesso." << std::endl;
    return 0;
