#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void trans(float *a, float *b, int l, int c){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < l and j < c){
        b[j * l + i] = a[i * c + j];
    }

}

int main(){
    int n, m;
    cin >> n >> m;
    int size = n * m * sizeof(int);
    int *a = new int[size];
    int *b = new int[size];

    for(int i=0; i < n; i++){
        for(int j=0; j < m; j++){
            int x; cin >> x;
            a[i * m + j] = x;
        }
    }

    int *da, *db;
    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    
    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);

    dim3 tpb(16, 16);
    dim3 bloc((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    trans<<<bloc, tbl>>>(da, db, n, m);
    cudaDeviceSynchronize();

    cudaMemcpy(b, db, size, cudaMemcpyDeviceToHost);

    for(int i=0; i < m; i++){
        for(int j=0; j < n; j++){
            cout << b[i * n + j] << " ";
        }
        cout << endl;
    }

    delete[] a;
    delete[] b;
    cudaFree(a);
    cudaFree(b);
    
    return 0;

}
