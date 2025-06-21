/* vrlsnmr: cpu implementation of kernel. */

namespace vrlsnmr::cpu {

/* weights: (n,)
 * ids: (m,)
 * tau: ()
 */
torch::Tensor kernel(
  torch::Tensor weights,
  torch::Tensor ids,
  torch::Tensor tau
) {
  using cfloat = c10::complex<float>;

  const int64_t n = weights.size(/*dim=*/0);
  const int64_t m = ids.size(/*dim=*/0);

  const auto coeffs = torch::fft::fft(weights.reciprocal()).conj() / n;
  const auto tau_inv = tau.reciprocal().item<float>();

  auto out = torch::empty({m, m}, /*options=*/coeffs.options());
  const auto coeffs_accessor = coeffs.accessor<cfloat, 1>();
  const auto ids_accessor = ids.accessor<int64_t, 1>();
  auto out_accessor = out.accessor<cfloat, 2>();

  for (int64_t i = 0; i < m; ++i) {
    out_accessor[i][i] = coeffs_accessor[0] + tau_inv;

    for (int64_t j = i + 1; j < m; ++j) {
      const auto k = n + ids_accessor[i] - ids_accessor[j];
      const auto Kij = coeffs_accessor[k];

      out_accessor[i][j] = Kij;
      out_accessor[j][i] = std::conj(Kij);
    }
  }

  return out;
}

} /* namespace vrlsnmr::cpu */
