/* vrlsnmr: cpu implementation of xmarginal_nd. */

namespace vrlsnmr::cpu {

namespace impl {

template<typename scalar_t>
void xmarginal_nd(
  const Grid grid,
  const accessor_t<c10::complex<scalar_t>, 4> kernel_inv,
  const accessor_t<scalar_t, 2> weights,
  const accessor_t<int64_t, 2> ids,
  accessor_t<scalar_t, 2> out
) {
  const int64_t bs = kernel_inv.size(/*dim=*/0);
  const int64_t nch = kernel_inv.size(/*dim=*/1);
  const int64_t m = kernel_inv.size(/*dim=*/2);

  #pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < bs; ++b) {
    for (int64_t k = 0; k < grid.size; ++k) {
      int64_t index[Grid::max_ndim];
      for (int d = 0; d < grid.ndim; ++d)
        index[d] = (k / grid.stride[d]) % grid.shape[d];

      const auto wk = weights[b][k];
      scalar_t sum = 0;

      for (int64_t c = 0; c < nch; ++c) {
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t j = 0; j < m; ++j) {
            scalar_t phase = 0;
            for (int d = 0; d < grid.ndim; ++d) {
              int64_t delta = ids[j][d] - ids[i][d];
              if (grid.flipped(c, d))
                delta = -delta;

              int64_t residue = (index[d] * delta) % grid.shape[d];
              residue = residue < 0 ? residue + grid.shape[d] : residue;
              phase += scalar_t(residue) / scalar_t(grid.shape[d]);
            }

            const auto theta = 2 * c10::pi<scalar_t> * phase;
            const c10::complex<scalar_t> alpha{std::cos(theta), std::sin(theta)};

            sum += (kernel_inv[b][c][i][j] * alpha).real();
          }
        }
      }

      out[b][k] = 1 / wk - sum / (nch * grid.size * wk * wk);
    }
  }
}

} /* namespace vrlsnmr::cpu::impl */

/* kernel_inv: (bs, c, m, m), where c = 2^(d-1)
 * weights: (bs, n_1, ..., n_d)
 * ids: (m, d)
 * out => (bs, n_1, ..., n_d)
 */
torch::Tensor xmarginal_nd(
  const torch::Tensor& kernel_inv,
  const torch::Tensor& weights,
  const torch::Tensor& ids
) {
  const Grid grid{weights};
  const int64_t bs = weights.size(/*dim=*/0);

  auto flat_weights = weights.contiguous().view({bs, grid.size});
  auto out = torch::empty({bs, grid.size}, /*options=*/weights.options());

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "xmarginal_nd", ([&] {
    impl::xmarginal_nd<scalar_t>(
      grid,
      access_as<c10::complex<scalar_t>, 4>(kernel_inv),
      access_as<scalar_t, 2>(flat_weights),
      access_as<int64_t, 2>(ids),
      access_as<scalar_t, 2>(out)
    );
  }));

  return out.view(weights.sizes());;
}

} /* namespace vrlsnmr::cpu */
