import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import load_inline, verify_ninja_availability

verify_ninja_availability()

with open("main.cu", "rt") as f:
    cuda_source = f.read()

module = load_inline(
    name='example',
    cpp_sources=[],
    cuda_sources=[cuda_source],
)

x = torch.arange(10, dtype=torch.float).to('cuda')

print(torch.ops.example.mymuladd.default(x, x, 0.0))
