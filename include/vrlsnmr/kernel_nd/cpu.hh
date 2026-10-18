/* vrlsnmr: cpu implementation of kernel_nd. */

namespace vrlsnmr::cpu {

namespace impl {

template<typename scalar_t>
void kernel_nd(
  const Grid grid,
  const accessor_t<c10::complex<scalar_t>, 2> coeffs,
  const accessor_t<scalar_t, 1> tau,
  const accessor_t<int64_t, 2> ids,
  accessor_t<c10::complex<scalar_t>, 4> out
) {
  const int64_t bs = out.size(/*dim=*/0);
  const int64_t nch = out.size(/*dim=*/1);
  const int64_t m = out.size(/*dim=*/2);

  #pragma omp parallel for collapse(3)
  for (int64_t b = 0; b < bs; ++b) {
    for (int64_t c = 0; c < nch; ++c) {
      for (int64_t i = 0; i < m; ++i) {
        const c10::complex<scalar_t> c0{coeffs[b][0].real()};
        out[b][c][i][i] = c0 + 1 / tau[b];

        for (int64_t j = i + 1; j < m; ++j) {
          int64_t delta[Grid::max_ndim];
          for (int d = 0; d < grid.ndim; ++d)
            delta[d] = ids[i][d] - ids[j][d];

          const int64_t idx = grid.index(c, delta);
          const auto Kij = coeffs[b][idx];

          out[b][c][i][j] = Kij;
          out[b][c][j][i] = std::conj(Kij);
        }
      }
    }
  }
}

} /* namespace vrlsnmr::cpu::impl */

/* weights: (bs, n_1, ..., n_d)
 * tau: (bs,)
 * ids: (m, d)
 * out => (bs, c, m, m), where c = 2^(d-1)
 */
torch::Tensor kernel_nd(
  const torch::Tensor& weights,
  const torch::Tensor& tau,
  const torch::Tensor& ids
) {
  const Grid grid{weights};

  const int64_t bs = weights.size(/*dim=*/0);
  const int64_t nch = grid.num_channels();
  const int64_t m = ids.size(/*dim=*/0);

  const auto coeffs = grid.make_coeffs(weights);
  auto out = torch::empty({bs, nch, m, m}, /*options=*/coeffs.options());

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "kernel_nd", ([&] {
      impl::kernel_nd<scalar_t>(
        grid,
        access_as<c10::complex<scalar_t>, 2>(coeffs),
        access_as<scalar_t, 1>(tau),
        access_as<int64_t, 2>(ids),
        access_as<c10::complex<scalar_t>, 4>(out)
      );
  }));

  return out;
}

} /* namespace vrlsnmr::cpu */
