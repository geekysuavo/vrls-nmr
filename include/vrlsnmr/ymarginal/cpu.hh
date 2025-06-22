/* vrlsnmr: cpu implementation of ymarginal. */

namespace vrlsnmr::cpu {

namespace impl {

template<typename scalar_t>
void ymarginal(
  const accessor_t<c10::complex<scalar_t>, 3> kernel_inv,
  const accessor_t<c10::complex<scalar_t>, 2> coeffs,
  const accessor_t<int64_t, 1> ids,
  accessor_t<scalar_t, 2> out
) {
  const int64_t bs = coeffs.size(/*dim=*/0);
  const int64_t n = coeffs.size(/*dim=*/1);
  const int64_t m = ids.size(/*dim=*/0);

  #pragma omp parallel for
  for (int64_t b = 0; b < bs; ++b) {
    for (int64_t k = 0; k < n; ++k) {
      scalar_t sum = 0;

      for (int64_t i = 0; i < m; ++i) {
        const int64_t index_i = ids[i] >= k ? ids[i] - k : n + ids[i] - k;
        const auto beta_i = std::conj(coeffs[b][index_i]);

        for (int64_t j = 0; j < m; ++j) {
          const int64_t index_j = ids[j] >= k? ids[j] - k : n + ids[j] - k;
          const auto beta_j = coeffs[b][index_j];

          sum += (kernel_inv[b][i][j] * beta_i * beta_j).real();
        }
      }

      out[b][k] = coeffs[b][0].real() - sum;
    }
  }
}

} /* namespace vrlsnmr::cpu::impl */

/* kernel_inv: (bs, m, m)
 * weights: (bs, n)
 * ids: (m,)
 * out => (bs, n)
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

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "ymarginal", ([&] {
    impl::ymarginal<scalar_t>(
      access_as<c10::complex<scalar_t>, 3>(kernel_inv),
      access_as<c10::complex<scalar_t>, 2>(coeffs),
      access_as<int64_t, 1>(ids),
      access_as<scalar_t, 2>(out)
    );
  }));

  return out;
}

} /* namespace vrlsnmr::cpu */
