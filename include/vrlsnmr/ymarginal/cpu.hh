/* vrlsnmr: cpu implementation of ymarginal. */

namespace vrlsnmr::cpu {

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
  const auto kernel_inv_accessor = kernel_inv.accessor<cfloat, 2>();
  const auto coeffs_accessor = coeffs.accessor<cfloat, 1>();
  const auto ids_accessor = ids.accessor<int64_t, 1>();
  auto out_accessor = out.accessor<float, 1>();

  for (int64_t k = 0; k < n; ++k) {
    float sum = 0.0f;

    for (int64_t i = 0; i < m; ++i) {
      const int64_t index_i = ids_accessor[i] >= k
                            ? ids_accessor[i] - k
                            : n + ids_accessor[i] - k;

      const cfloat beta_i = std::conj(coeffs_accessor[index_i]);

      for (int64_t j = 0; j < m; ++j) {
        const int64_t index_j = ids_accessor[j] >= k
                              ? ids_accessor[j] - k
                              : n + ids_accessor[j] - k;

        const cfloat beta_j = coeffs_accessor[index_j];

        sum += (kernel_inv_accessor[i][j] * beta_i * beta_j).real();
      }
    }

    out_accessor[k] = coeffs_accessor[0].real() - sum;
  }

  return out;
}

} /* namespace vrlsnmr::cpu */
