/* vrlsnmr: cuda implementation of kernel. */

namespace vrlsnmr::cuda {

namespace impl {

template<typename scalar_t>
__global__ void kernel(
  const accessor_t<c10::complex<scalar_t>, 2> coeffs,
  const accessor_t<scalar_t, 1> tau,
  const accessor_t<int64_t, 1> ids,
  accessor_t<c10::complex<scalar_t>, 3> out
) {
  const int64_t bs = coeffs.size(/*dim=*/0);
  const int64_t n = coeffs.size(/*dim=*/1);
  const int64_t m = ids.size(/*dim=*/0);

  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  if (i >= m || j >= m)
    return;

  const int64_t ids_i = ids[i];
  const int64_t ids_j = ids[j];

  for (int64_t b = blockIdx.z; b < bs; b += gridDim.z) {
    c10::complex<scalar_t> elem;

    if (i == j) {
      elem = c10::complex<scalar_t>{coeffs[b][0].real() + 1 / tau[b], 0};
    }
    else if (i < j) {
      const int64_t diff = ids_i - ids_j;
      const int64_t k = diff >= 0 ? diff : n + diff;
      elem = coeffs[b][k];
    }
    else {
      const int64_t diff = ids_j - ids_i;
      const int64_t k = diff >= 0 ? diff : n + diff;
      elem = std::conj(coeffs[b][k]);
    }

    out[b][i][j] = elem;
  }
}

} /* namespace vrlsnmr::cuda::impl */

/* weights: (bs, n)
 * tau: (bs,)
 * ids: (m,)
 * out => (bs, m, m)
 */
torch::Tensor kernel(
  const torch::Tensor& weights,
  const torch::Tensor& tau,
  const torch::Tensor& ids
) {
  const int64_t bs = weights.size(/*dim=*/0);
  const int64_t n = weights.size(/*dim=*/1);
  const int64_t m = ids.size(/*dim=*/0);

  const auto coeffs = torch::fft::fft(weights.reciprocal()).conj() / n;
  auto out = torch::empty({bs, m, m}, /*options=*/coeffs.options());

  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 threads(32, 8);
  const dim3 blocks(
    (m + threads.x - 1) / threads.x,
    (m + threads.y - 1) / threads.y,
    std::min(bs, max_grid_yz)
  );

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "kernel", ([&] {
    impl::kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      access_as<c10::complex<scalar_t>, 2>(coeffs),
      access_as<scalar_t, 1>(tau),
      access_as<int64_t, 1>(ids),
      access_as<c10::complex<scalar_t>, 3>(out)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }));

  return out;
}

} /* namespace vrlsnmr::cuda */
