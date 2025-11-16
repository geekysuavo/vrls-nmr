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
  const int bs = coeffs.size(/*dim=*/0);
  const int n = coeffs.size(/*dim=*/1);
  const int m = ids.size(/*dim=*/0);

  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int ij = blockIdx.y * blockDim.y + threadIdx.y;

  if (b < bs && ij < m * m) {
    const int i = ij / m;
    const int j = ij - i * m;

    if (i == j) {
      const c10::complex<scalar_t> c0{coeffs[b][0].real()};
      out[b][i][j] = c0 + 1 / tau[b];
    }
    else if (i < j) {
      const int k = n + ids[i] - ids[j];
      const auto Kij = coeffs[b][k];

      out[b][i][j] = Kij;
      out[b][j][i] = std::conj(Kij);
    }
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

  const dim3 threads(32, 32);
  const dim3 blocks(
    (bs + threads.x - 1) / threads.x,
    (m * m + threads.y - 1) / threads.y
  );

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "kernel", ([&] {
    impl::kernel<scalar_t><<<blocks, threads>>>(
      access_as<c10::complex<scalar_t>, 2>(coeffs),
      access_as<scalar_t, 1>(tau),
      access_as<int64_t, 1>(ids),
      access_as<c10::complex<scalar_t>, 3>(out)
    );
  }));

  return out;
}

} /* namespace vrlsnmr::cuda */
