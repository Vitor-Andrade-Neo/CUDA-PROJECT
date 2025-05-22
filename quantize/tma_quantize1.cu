#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

namespace py = pybind11;
using namespace cute;

#define FINAL_MASK 0xffffffff
#define WARP_SIZE 32
constexpr int CTA_M = 32;
constexpr int CTA_N = 32;

// Kernel TMA para quantização 
template<class TmaLoad, class TmaStoreQuant, class TmaStoreScales>
__global__ void tma_mxfp8_full_quantize_kernel(
    __grid_constant__ const TmaLoad tma_load,
    __grid_constant__ const TmaStoreQuant tma_store_quant,
    __grid_constant__ const TmaStoreScales tma_store_scales,
    auto input_tensor, auto quantized_tensor, auto scales_tensor,
    int M, int N) {
    using namespace cute;

    // SMEM buffers
    __shared__ half input_smem_data[CTA_M * CTA_N];
    __shared__ cutlass::float_ue4m3_t quantized_smem_data[CTA_M * CTA_N];
    __shared__ cutlass::float_ue8m0_t scales_smem_data[CTA_M];
    __shared__ uint64_t tma_load_mbar;

    // Layouts SMEM
    auto input_smem_layout     = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
    auto quantized_smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
    auto scales_smem_layout    = make_layout(make_shape(CTA_M, Int<1>{}), LayoutRight{});

    auto input_smem_tensor     = make_tensor(make_smem_ptr(input_smem_data),     input_smem_layout);
    auto quantized_smem_tensor = make_tensor(make_smem_ptr(quantized_smem_data), quantized_smem_layout);
    auto scales_smem_tensor    = make_tensor(make_smem_ptr(scales_smem_data),    scales_smem_layout);

    int block_m = blockIdx.y;
    int block_n = blockIdx.z;
    if (block_m * CTA_M >= M || block_n * CTA_N >= N) return;

    // FASE 1: TMA LOAD 
    if (threadIdx.x == 0) {
        auto coord_in     = tma_load.get_tma_tensor(shape(input_tensor));
        auto coord_in_cta = local_tile(coord_in, Tile<Int<CTA_M>, Int<CTA_N>>{}, make_coord(block_m, block_n));
        constexpr int bytes = CTA_M * CTA_N * sizeof(half);
        initialize_barrier(tma_load_mbar, 1);
        set_barrier_transaction_bytes(tma_load_mbar, bytes);

        auto slice = tma_load.get_slice(0);
        copy(tma_load.with(tma_load_mbar), slice.partition_S(coord_in_cta), slice.partition_D(input_smem_tensor));
    }
    __syncthreads();
    wait_barrier(tma_load_mbar, 0);

    // FASE 2: ESCALAS E QUANTIZAÇÃO
    int tid      = threadIdx.x;
    int warp_id  = tid / WARP_SIZE;
    int lane_id  = tid % WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // 2.1 Cálculo de escalas por linha
    for (int row = warp_id; row < CTA_M; row += num_warps) {
        int global_row = block_m * CTA_M + row;
        if (global_row >= M) continue;

        half local_val = __float2half(0.0f);
        if (lane_id < CTA_N) {
            int global_col = block_n * CTA_N + lane_id;
            if (global_col < N) local_val = input_smem_data[row * CTA_N + lane_id];
        }
        half max_val = __habs(local_val);
        for (int off = 16; off > 0; off /= 2) {
            half sh = __shfl_xor_sync(FINAL_MASK, max_val, off, 32);
            max_val = __hmax(__habs(max_val), __habs(sh));
        }
        if (lane_id == 0) {
            float max_f = __half2float(max_val);
            cutlass::float_ue8m0_t sc;
            cutlass::NumericConverter<cutlass::float_ue8m0_t, float> cnv;
            sc = (max_f != 0.0f) ? cnv(225.0f / fabsf(max_f)) : cnv(0.0f);
            scales_smem_data[row] = sc;
        }
    }
    __syncthreads();

    // 2.2 Quantização usando as escalas
    for (int idx = tid; idx < CTA_M * CTA_N; idx += blockDim.x) {
        int row = idx / CTA_N;
        int col = idx % CTA_N;
        int gr = block_m * CTA_M + row;
        int gc = block_n * CTA_N + col;
        if (gr < M && gc < N) {
            half in_h = input_smem_data[row * CTA_N + col];
            cutlass::float_ue8m0_t sc = scales_smem_data[row];
            float f_in = __half2float(in_h);
            float f_sc; cutlass::NumericConverter<float, cutlass::float_ue8m0_t> sc_cnv;
            f_sc = sc_cnv(sc);
            float scaled = f_in * f_sc;
            cutlass::NumericConverter<cutlass::float_ue4m3_t, float> q_cnv;
            quantized_smem_data[row * CTA_N + col] = q_cnv(scaled);
        }
    }
    __syncthreads();

    // FASE 3: TMA STORE
    tma_store_fence();
    if (threadIdx.x == 0) {
        // store quantizado
        auto coord_q    = tma_store_quant.get_tma_tensor(shape(quantized_tensor));
        auto coord_q_cta= local_tile(coord_q, Tile<Int<CTA_M>, Int<CTA_N>>{}, make_coord(block_m, block_n));
        auto slice_q    = tma_store_quant.get_slice(0);
        copy(tma_store_quant, slice_q.partition_S(quantized_smem_tensor), slice_q.partition_D(coord_q_cta));

        // store escalas
        auto coord_s     = tma_store_scales.get_tma_tensor(shape(scales_tensor));
        auto coord_s_cta = local_tile(coord_s, Tile<Int<CTA_M>, Int<1>>{}, make_coord(block_m, Int<0>{}));
        auto slice_s     = tma_store_scales.get_slice(0);
        copy(tma_store_scales, slice_s.partition_S(scales_smem_tensor), slice_s.partition_D(coord_s_cta));
    }
    __syncthreads();
}

// Host: quantização TMA MXFP8 completa tma_mxfp8_quantize(torch::Tensor input_tensor) {
    using namespace cute;
    auto sizes = input_tensor.sizes();
    int M = sizes[0], N = sizes[1];

    // Tensores saída: float32 proxies
    auto opts_sc = torch::TensorOptions().dtype(torch::kFloat32).device(input_tensor.device());
    auto opts_q  = torch::TensorOptions().dtype(torch::kFloat32).device(input_tensor.device());
    torch::Tensor scales_tensor    = torch::zeros({M,1}, opts_sc);
    torch::Tensor quantized_tensor = torch::zeros({M,N}, opts_q);

    // Ponteiros
    half* input_ptr = reinterpret_cast<half*>(input_tensor.data_ptr<torch::Half>());
    auto scales_ptr = reinterpret_cast<cutlass::float_ue8m0_t*>(scales_tensor.data_ptr<float>());
    auto quant_ptr  = reinterpret_cast<cutlass::float_ue4m3_t*>(quantized_tensor.data_ptr<float>());

    // Layouts GMEM
    auto in_layout  = make_layout(make_shape(M,N), LayoutRight{});
    auto sc_layout  = make_layout(make_shape(M,1), LayoutRight{});
    auto q_layout   = make_layout(make_shape(M,N), LayoutRight{});
    auto in_gmem    = make_tensor(make_gmem_ptr(input_ptr), in_layout);
    auto sc_gmem    = make_tensor(make_gmem_ptr(scales_ptr), sc_layout);
    auto q_gmem     = make_tensor(make_gmem_ptr(quant_ptr), q_layout);

    // SMEM layouts
    auto in_smem_l  = make_layout(make_shape(CTA_M,CTA_N), LayoutRight{});
    auto sc_smem_l  = make_layout(make_shape(CTA_M,1), LayoutRight{});
    auto q_smem_l   = make_layout(make_shape(CTA_M,CTA_N), LayoutRight{});

    // Objetos TMA
    auto tma_load      = make_tma_copy(SM90_TMA_LOAD{},  in_gmem, in_smem_l);
    auto tma_store_q   = make_tma_copy(SM90_TMA_STORE{}, q_gmem,   q_smem_l);
    auto tma_store_sc  = make_tma_copy(SM90_TMA_STORE{}, sc_gmem,  sc_smem_l);

    // Grid & block
    dim3 grid(1, (M+CTA_M-1)/CTA_M, (N+CTA_N-1)/CTA_N);
    dim3 block(CTA_M*CTA_N);

    // Lançamento
    tma_mxfp8_full_quantize_kernel<<<grid, block>>>(
        tma_load, tma_store_q, tma_store_sc,
        in_gmem, q_gmem, sc_gmem,
        M, N
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Erro TMA MXFP8: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    // Impressão de verificação
    auto scales_cpu = scales_tensor.cpu();
    float* sc_data = scales_cpu.data_ptr<float>();
    printf("Primeiras escalas:\n");
    for (int i=0; i<min(10,M); i++) {
        auto ptr = reinterpret_cast<cutlass::float_ue8m0_t*>(&sc_data[i]);
        printf("Scale[%d] = %f\n", i, *ptr);
    }

PYBIND11_MODULE(tma_neoQuant, m) {
    m.doc() = "TMA-based MXFP8 Quantization (full only)";
    m.def("tma_quantize", &tma_mxfp8_quantize,
          "Full MXFP8 quantization using TMA Load/Store",
          py::arg("torch_tensor"));
}
