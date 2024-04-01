[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void compute_weights(
    const DiffTensorView<float> sigmas,
    DiffTensorView<float> weights,
    no_diff const TensorView<float> steps,
    no_diff const TensorView<int> info,
    no_diff const int n_rays)
{

    const int idx = (cudaThreadIdx() + cudaBlockDim() * cudaBlockIdx()).x;
    if (idx >= n_rays) return;

    const int n_samples = info[idx, 1];
    const int ray_start = info[idx, 0];
    const int ray_end = ray_start + n_samples;

    if (n_samples == 0) return;

    float transmittance = 1.;
    float alpha;
    int k = ray_start;

    [MaxIters(400)]
    while (k < ray_end) {
        alpha = exp(-sigmas[k] * no_diff steps[k]);
        weights[k] = transmittance * (1. - alpha);
        transmittance *= alpha;
        k++;
    }
    return;
}

