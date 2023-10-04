import torch
import torch.nn as nn
from awq.modules.linear import WQLinear_GEMM, WQLinear_TORCH

def torch_only():
    in_features = 128
    out_features = 64
    w_bit = 4
    group_size = 128
    linear = nn.Linear(in_features, out_features, bias=False)
    scales = torch.rand(in_features // group_size, out_features)
    zeros = torch.rand(in_features // group_size, out_features)
    x = torch.randn(1, in_features)

    wq_linear = WQLinear_GEMM.from_linear(linear, w_bit, group_size, scales=scales, zeros=zeros)
    wq_linear_torch = WQLinear_TORCH.from_wqlinear_gemm(wq_linear)

    with torch.no_grad():
        out = linear(x)
        dq_out = wq_linear_torch(x)

        print(out)
        print(dq_out)

if __name__ == '__main__':
    torch_only()