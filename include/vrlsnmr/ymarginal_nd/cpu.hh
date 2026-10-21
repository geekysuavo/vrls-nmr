/* vrlsnmr: cpu implementation of ymarginal_nd. */

namespace vrlsnmr::cpu {

namespace impl {

template<typename scalar_t>
void ymarginal_nd(
  const Grid grid,
  const Grid grid_out,
  const accessor_t<c10::complex<scalar_t>, 4> kernel_inv,
  const accessor_t<c10::complex<scalar_t>, 2> coeffs,
  const accessor_t<int64_t, 2> ids,
  accessor_t<scalar_t, 2> out
) {
  const int64_t bs = kernel_inv.size(/*dim=*/0);
  const int64_t nch = kernel_inv.size(/*dim=*/1);
  const int64_t m = kernel_inv.size(/*dim=*/2);

  #pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < bs; ++b) {
    for (int64_t k = 0; k < grid_out.size; ++k) {
      int64_t index[Grid::max_ndim];
      for (int d = 0; d < grid_out.ndim; ++d)
        index[d] = (k / grid_out.stride[d]) % grid_out.shape[d];

      scalar_t sum = 0;

      for (int64_t c = 0; c < nch; ++c) {
        for (int64_t i = 0; i < m; ++i) {
          int64_t delta[Grid::max_ndim];
          for (int d = 0; d < grid.ndim; ++d)
            delta[d] = ids[i][d] - index[d];

          const int64_t idx_i = grid.index(c, delta);
          const auto beta_i = std::conj(coeffs[b][idx_i]);

          for (int64_t j = 0; j < m; ++j) {
            for (int d = 0; d < grid.ndim; ++d)
              delta[d] = ids[j][d] - index[d];

            const int64_t idx_j = grid.index(c, delta);
            const auto beta_j = coeffs[b][idx_j];

            sum += (kernel_inv[b][c][i][j] * beta_i * beta_j).real();
          }
        }
      }

      out[b][k] = coeffs[b][0].real() - sum / nch;
    }
  }
}

} /* namespace vrlsnmr::cpu::impl */

/* kernel_inv: (bs, c, m, m), where c=2^(d-1)
 * weights: (bs, n_1, ..., n_d)
 * ids: (m, d)
 * out => (bs, n_1, ..., n_d) or (bs, n_1 / 2, ..., n_d / 2)
 */
torch::Tensor ymarginal_nd(
  const torch::Tensor& kernel_inv,
  const torch::Tensor& weights,
  const torch::Tensor& ids,
  const bool windowed
) {
  const Grid grid{weights};
  const Grid grid_out = windowed ? grid.half() : grid;

  const int64_t bs = weights.size(/*dim=*/0);

  auto coeffs = grid.make_coeffs(weights);
  auto out = torch::empty({bs, grid_out.size}, /*options=*/weights.options());

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "ymarginal_nd", ([&] {
    impl::ymarginal_nd<scalar_t>(
      grid,
      grid_out,
      access_as<c10::complex<scalar_t>, 4>(kernel_inv),
      access_as<c10::complex<scalar_t>, 2>(coeffs),
      access_as<int64_t, 2>(ids),
      access_as<scalar_t, 2>(out)
    );
  }));

  std::vector<int64_t> shape{weights.sizes().vec()};
  for (int d = 0; d < grid_out.ndim; ++d)
    shape[d + 1] = grid_out.shape[d];

  return out.view(shape);
}

} /* namespace vrlsnmr::cpu */
