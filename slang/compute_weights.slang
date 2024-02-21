[AutoPyBindCUDA]
[CUDAKernel]
void compute_weights_fwd(
    const TensorView<float> sigmas,
    const TensorView<float> steps,
    const TensorView<int> info,
    const float threshold,
    TensorView<float> weights,
    const int n_rays)
{
    
    const int idx = cudaThreadIdx().x + cudaBlockDim().x * cudaBlockIdx().x;
    if (idx >= n_rays) return;

    const int n_samples = info[2 * idx + 1];
    const int ray_start = info[2 * idx];
    const int ray_end = ray_start + n_samples;
    if (n_samples == 0) return;

    float transmittance = 1.;
    float alpha;
    int k = ray_start;
    int loops = 0; 
    // early terminate ray if transmittance under threshold
    while (transmittance > threshold && k < ray_end) {
        alpha = exp(-sigmas[k] * steps[k]);
        weights[k] = transmittance * (1. - alpha);
        transmittance *= alpha;
        k++;
        loops++;
    }
    printf("%i", loops);
    return;
}

[AutoPyBindCUDA]
[CUDAKernel]
void compute_weights_bwd(
    const TensorView<float> sigmas,
    const TensorView<float> steps,
    const TensorView<int> info,
    TensorView<float> weights,
    const TensorView<float> grad_weights,
    TensorView<float> grad_sigmas,
    const int n_rays
) {
    const int idx = cudaThreadIdx().x + cudaBlockDim().x * cudaBlockIdx().x;
    if (idx >= n_rays) return;

    const int n_samples = info[2 * idx + 1];
    const int ray_start = info[2 * idx];
    const int ray_end = ray_start + n_samples;
    if (n_samples == 0) return;

    float acc = 0.;
    float transmittance = 1.;
    for (int k = ray_start; k < ray_end; k++) acc -= weights[k] * grad_weights[k];
    for (int k = ray_start; k < ray_end; k++) {
        acc += weights[k] * grad_weights[k];
        transmittance *= exp(-sigmas[k] * steps[k]);
        grad_sigmas[k] = steps[k] * (acc + transmittance * grad_weights[k]);
    }
    return;
}

