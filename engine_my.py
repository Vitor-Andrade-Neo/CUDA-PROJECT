import torch
from transformer_engine.pytorch import fp8_autocast, gemm
from transformer_engine.pytorch.fp8 import cast_to_fp8, cast_from_fp8
from transformer_engine.common.recipe import DelayedScaling, Format
device = 'cuda'
M, K, N = 128, 256, 64
# Tensors FP32
A_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
B_fp32 = torch.randn(K, N, device=device, dtype=torch.float32)
# Configuração de FP8 com DelayedScaling
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,  # E4M3 para forward (ativações), E5M2 para backward
    amax_history_len=16,       # suaviza mudanças de escala
    amax_compute_algo="max",   # calcula amax como máximo absoluto
    scaling_factor_compute_algo="max",  # escala baseada no amax máximo
)
# Buffers auxiliares
A_fp8 = torch.empty_like(A_fp32, dtype=torch.uint8)
B_fp8 = torch.empty_like(B_fp32, dtype=torch.uint8)
A_scale = torch.empty(1, device=device)
A_scale_inv = torch.empty(1, device=device)
A_amax = torch.empty(1, device=device)
B_scale = torch.empty(1, device=device)
B_scale_inv = torch.empty(1, device=device)
B_amax = torch.empty(1, device=device)
#Cria as versões quantizadas em FP8, cria as escalas, escalas inversas para dequantizaçãoe e amax/bmax
# Quantização FP8
cast_to_fp8(A_fp32, A_fp8, A_scale, A_scale_inv, A_amax, 0, fp8_recipe, forward=True)
cast_to_fp8(B_fp32, B_fp8, B_scale, B_scale_inv, B_amax, 1, fp8_recipe, forward=True)
# Tensor de saída (FP32)
C_fp32 = torch.empty(M, N, device=device, dtype=torch.float32)
# MatMul com acumulação em FP32
with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    C = gemm(A_fp32, B_fp32)  # A e B serão quantizados usando recipe + escalas
    print("C shape:", C.shape)
    print("C dtype:", C.dtype)
# Resultado está em FP32
