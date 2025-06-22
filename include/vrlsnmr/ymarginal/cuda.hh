/* vrlsnmr: cuda implementation of ymarginal. */

namespace vrlsnmr::cuda {

namespace impl {

template<typename scalar_t>
__global__ void ymarginal(
  const accessor_t<c10::complex<scalar_t>, 3> kernel_inv,
  const accessor_t<c10::complex<scalar_t>, 2> coeffs,
  const accessor_t<int64_t, 1> ids,
  accessor_t<scalar_t, 2> out
) {
  const int bs = coeffs.size(/*dim=*/0);
  const int n = coeffs.size(/*dim=*/1);
  const int m = ids.size(/*dim=*/0);

  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (b < bs && k < n) {
    scalar_t sum = 0;

    for (int i = 0; i < m; ++i) {
      const int index_i = ids[i] >= k ? ids[i] - k : n + ids[i] - k;
      const auto beta_i = std::conj(coeffs[b][index_i]);

      for (int j = 0; j < m; ++j) {
        const int index_j = ids[j] >= k ? ids[j] - k : n + ids[j] - k;
        const auto beta_j = coeffs[b][index_j];

        sum += (kernel_inv[b][i][j] * beta_i * beta_j).real();
      }
    }

    out[b][k] = coeffs[b][0].real() - sum;
  }
}

} /* namespace vrlsnmr::cuda::impl */

/* kernel_inv: (b, m, m)
 * weights: (b, n)
 * ids: (m,)
 * out => (b, n)
 */
torch::Tensor ymarginal(
  const torch::Tensor& kernel_inv,
  const torch::Tensor& weights,
  const torch::Tensor& ids
) {
  const int64_t bs = weights.size(/*dim=*/0);
  const int64_t n = weights.size(/*dim=*/1);

  const auto coeffs = torch::fft::fft(weights.reciprocal()).conj() / n;
  auto out = torch::empty({bs, n}, /*options=*/weights.options());

  const dim3 threads(32, 32);
  const dim3 blocks(
    (bs + threads.x - 1) / threads.x,
    (n + threads.y - 1) / threads.y
  );

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "ymarginal", ([&] {
    impl::ymarginal<scalar_t><<<blocks, threads>>>(
      access_as<c10::complex<scalar_t>, 3>(kernel_inv),
      access_as<c10::complex<scalar_t>, 2>(coeffs),
      access_as<int64_t, 1>(ids),
      access_as<scalar_t, 2>(out)
    );
  }));

  return out;
}

} /* namespace vrlsnmr::cuda */
