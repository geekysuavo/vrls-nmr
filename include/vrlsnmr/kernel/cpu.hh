/* vrlsnmr: cpu implementation of kernel. */

namespace vrlsnmr::cpu {

namespace impl {

template<typename scalar_t>
void kernel(
  const accessor_t<c10::complex<scalar_t>, 2> coeffs,
  const accessor_t<scalar_t, 1> tau,
  const accessor_t<int64_t, 1> ids,
  accessor_t<c10::complex<scalar_t>, 3> out
) {
  const int64_t bs = coeffs.size(/*dim=*/0);
  const int64_t n = coeffs.size(/*dim=*/1);
  const int64_t m = ids.size(/*dim=*/0);

  #pragma omp parallel for
  for (int64_t b = 0; b < bs; ++b) {
    for (int64_t i = 0; i < m; ++i) {
      out[b][i][i] = coeffs[b][0] + 1 / tau[b];

      for (int64_t j = i + 1; j < m; ++j) {
        const auto k = n + ids[i] - ids[j];
        const auto Kij = coeffs[b][k];

        out[b][i][j] = Kij;
        out[b][j][i] = std::conj(Kij);
      }
    }
  }
}

} /* namespace vrlsnmr::cpu::impl */

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

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "kernel", ([&] {
      impl::kernel<scalar_t>(
        access_as<c10::complex<scalar_t>, 2>(coeffs),
        access_as<scalar_t, 1>(tau),
        access_as<int64_t, 1>(ids),
        access_as<c10::complex<scalar_t>, 3>(out)
      );
  }));

  return out;
}

} /* namespace vrlsnmr::cpu */
