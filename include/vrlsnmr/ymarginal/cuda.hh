/* vrlsnmr: cuda implementation of ymarginal. */

namespace vrlsnmr::cuda {

template<typename scalar_t>
__global__ void ymarginal_forward(
  const accessor_t<c10::complex<scalar_t>, 2> kernel_inv,
  const accessor_t<c10::complex<scalar_t>, 1> coeffs,
  const accessor_t<int64_t, 1> ids,
  accessor_t<scalar_t, 1> out
) {
  const int m = ids.size(/*dim=*/0);
  const int n = out.size(/*dim=*/0);

  const int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < n) {
    scalar_t sum = 0.0f;

    for (int i = 0; i < m; ++i) {
      const int index_i = ids[i] >= k ? ids[i] - k : n + ids[i] - k;
      const auto beta_i = std::conj(coeffs[index_i]);

      for (int j = 0; j < m; ++j) {
        const int index_j = ids[j] >= k ? ids[j] - k : n + ids[j] - k;
        const auto beta_j = coeffs[index_j];

        sum += (kernel_inv[i][j] * beta_i * beta_j).real();
      }
    }

    out[k] = coeffs[0].real() - sum;
  }
}

/* kernel_inv: (m, m)
 * weights: (n,)
 * ids: (m,)
 */
torch::Tensor ymarginal(
  torch::Tensor kernel_inv,
  torch::Tensor weights,
  torch::Tensor ids
) {
  using cfloat = c10::complex<float>;

  const int64_t n = weights.size(/*dim=*/0);
  const int64_t m = ids.size(/*dim=*/0);

  const auto coeffs = torch::fft::fft(weights.reciprocal()).conj() / n;

  auto out = torch::empty({n}, /*options=*/weights.options());

  const int threads = 1024;
  const dim3 blocks((n + threads - 1) / threads);

  ymarginal_forward<float><<<blocks, threads>>>(
    access_as<cfloat, 2>(kernel_inv),
    access_as<cfloat, 1>(coeffs),
    access_as<int64_t, 1>(ids),
    access_as<float, 1>(out)
  );

  return out;
}

} /* namespace vrlsnmr::cuda */
