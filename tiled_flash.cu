#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;

#define BLOCK_SIZE 16 
#define D 64           

__global__ void flash(float *Q, float *K, float *V, float *O, int N) {
    __shared__ float tileQ[BLOCK_SIZE][D];  // Shared memory para Q
    __shared__ float tileK[BLOCK_SIZE][D];  // Shared memory para K
    __shared__ float tileV[BLOCK_SIZE][D];  // Shared memory para V

    int tx = threadIdx.x;
    int bx = blockIdx.x * BLOCK_SIZE;

    float od[D] = {0.0f};

    // Loop sobre os blocos de K e V
    for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
        // Carregar Q e K na shared memory
        for (int d = 0; d < D; d++) {
            tileQ[tx][d] = Q[(bx + tx) * D + d];
            tileK[tx][d] = K[(bk + tx) * D + d];
            tileV[tx][d] = V[(bk + tx) * D + d];
        }
        __syncthreads();  // Sincroniza os threads para garantir que os dados foram carregados

        // Computa a matriz de atenção QK^T
        float score[BLOCK_SIZE] = {0.0f};
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int d = 0; d < D; d++) {
                score[i] += tileQ[tx][d] * tileK[i][d];
            }
        }

        // Aplicação da normalização por sqrt(D)
        for (int i = 0; i < BLOCK_SIZE; i++) {
            score[i] /= sqrtf(D);
        }

        // Aplicação do softmax (versão simplificada)
        float max_score = -1e9;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            max_score = fmaxf(max_score, score[i]);
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            score[i] = expf(score[i] - max_score);
            sum_exp += score[i];
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
            score[i] /= sum_exp;
        }

        // Multiplicação do softmax com V
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int d = 0; d < D; d++) {
                od[d] += score[i] * tileV[i][d];
            }
        }
        __syncthreads();  // Sincroniza os threads antes do próximo bloco
    }

    // Escreve a saída de volta para a memória global
    for (int d = 0; d < D; d++) {
        O[(bx + tx) * D + d] = od[d];
    }
}

int main(){
    int N = 1024;
    float *hq = new float[N*D];
    float *hk = new float[N*D];
    float *hv = new float[N*D];
    float *ho = new float[N*D];

    for(int i=0; i<N; i++){
        hq[i] = static_cast<float>(rand()) / RAND_MAX;
        hk[i] = static_cast<float>(rand()) / RAND_MAX;    
        hv[i] = static_cast<float>(rand()) / RAND_MAX;    
        ho[i] = static_cast<float>(rand()) / RAND_MAX;    
    }
    
    float *dq, *dk, *dv, *od;

    cudaMalloc(&dq, N * D * sizeof(float));
    cudaMalloc(&dk, N * D * sizeof(float));
    cudaMalloc(&dv, N * D * sizeof(float));
    cudaMalloc(&od, N * D * sizeof(float));

    cudaMemcpy(dq, hq, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(od, ho, N*sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(numBlocks);
    dim3 block(BLOCK_SIZE);

    flash<<<grid, block>>>(dq, dk, dv, od, N);

    cudaFree(hq);
    cudaFree(hk);
    cudaFree(hv);
    cudaFree(ho);

    return 0;
}
