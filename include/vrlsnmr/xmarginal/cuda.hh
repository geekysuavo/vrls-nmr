/* vrlsnmr: cuda implementation of xmarginal. */

namespace vrlsnmr::cuda {

namespace impl {

template<typename scalar_t>
__global__ void xmarginal(
  const accessor_t<c10::complex<scalar_t>, 3> kernel_inv,
  const accessor_t<int64_t, 1> ids,
  accessor_t<scalar_t, 3> hist
) {
  const int64_t bs = hist.size(/*dim=*/0);
  const int64_t n = hist.size(/*dim=*/1);
  const int64_t m = ids.size(/*dim=*/0);

  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  if (i >= m || j >= m)
    return;

  int64_t diff = ids[j] - ids[i];
  diff = diff >= 0 ? diff : diff + n;

  for (int64_t b = blockIdx.z; b < bs; b += gridDim.z) {
    const auto elem = kernel_inv[b][i][j];
    atomicAdd(&hist[b][diff][0], elem.real());
    atomicAdd(&hist[b][diff][1], elem.imag());
  }
}

} /* namespace vrlsnmr::cuda::impl */

/* kernel_inv: (bs, m, m)
 * weights: (bs, n)
 * ids: (m,)
 * out => (bs, n)
 */
torch::Tensor xmarginal(
  const torch::Tensor& kernel_inv,
  const torch::Tensor& weights,
  const torch::Tensor& ids
) {
  const int64_t bs = weights.size(/*dim=*/0);
  const int64_t n = weights.size(/*dim=*/1);
  const int64_t m = ids.size(/*dim=*/0);

  auto hist = torch::zeros({bs, n}, /*options=*/kernel_inv.options());
  auto hist_real = torch::view_as_real(hist);

  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 threads(32, 8);
  const dim3 blocks(
    (m + threads.x - 1) / threads.x,
    (m + threads.y - 1) / threads.y,
    std::min(bs, max_grid_yz)
  );

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "xmarginal", ([&] {
    impl::xmarginal<scalar_t><<<blocks, threads, 0, stream>>>(
      access_as<c10::complex<scalar_t>, 3>(kernel_inv),
      access_as<int64_t, 1>(ids),
      access_as<scalar_t, 3>(hist_real)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }));

  const auto sums = torch::real(torch::fft::ifft(hist));
  return weights.reciprocal() - sums / weights.square();
}

} /* namespace vrlsnmr::cuda */
