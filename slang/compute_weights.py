import torch
import slangpy
from torch.utils.cpp_extension import load

_cuda = load(name="_cuda", sources=['../src/cuda.cu'], verbose=True, build_directory='../build')

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

# m = slangpy.loadModule('compute_weights.slang')

class NerfWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx: any, sigmas: torch.Tensor, steps: torch.Tensor, info: torch.Tensor, threshold: float) -> torch.Tensor: # type: ignore
        sigmas = sigmas.contiguous()
        steps = steps.contiguous()
        info = info.contiguous()
        weights = m.compute_weights(sigmas, steps, info, threshold)
        ctx.save_for_backward(sigmas, steps, info, weights)
        return weights 

    @staticmethod
    def backward(ctx: any, grad_weights: torch.Tensor): # type: ignore
        sigmas, steps, info, weights = ctx.saved_tensors
        grad_sigmas =weights = m.compute_weights.backward(sigmas, steps, info, weights, grad_weights) # type: ignore
        return grad_sigmas, None, None, None

#torch.Size([8636]) torch.Size([8636]) torch.Size([128, 2]) 0.0001 david choi

def main():
    sigmas = torch.tensor([1,2,3,4], dtype=torch.float).cuda()
    steps = torch.tensor([1,2,3,4], dtype=torch.float).cuda()
    info = torch.tensor([[4, 1]], dtype=torch.int).cuda()
    threshold = 0.0001
    weights_CUDA = NerfWeightsCUDA.apply(sigmas, steps, info, threshold)
    print(weights_CUDA)
    # weights_slang = NerfWeights.apply(sigmas, steps, info, threshold)
    

main()