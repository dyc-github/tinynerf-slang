import torch
import slangpy
from torch.utils.cpp_extension import load

_cuda = load(name="_cuda", sources=['./src/cuda.cu'], verbose=True, build_directory='./build')

class NerfWeightsCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx: any, sigmas: torch.Tensor, steps: torch.Tensor, info: torch.Tensor, threshold: float) -> torch.Tensor: # type: ignore
        sigmas = sigmas.contiguous()
        steps = steps.contiguous()
        info = info.contiguous()
        weights = _cuda.compute_weights_fwd(sigmas, steps, info, threshold) # type: ignore
        ctx.save_for_backward(sigmas, steps, info, weights)
        return weights 

    @staticmethod
    def backward(ctx: any, grad_weights: torch.Tensor): # type: ignore
        grad_weights = grad_weights.contiguous()
        sigmas, steps, info, weights = ctx.saved_tensors
        grad_sigmas = _cuda.compute_weights_bwd(sigmas, steps, info, weights, grad_weights) # type: ignore
        return grad_sigmas, None, None, None