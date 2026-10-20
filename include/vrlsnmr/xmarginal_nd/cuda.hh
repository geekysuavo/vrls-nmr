/* vrlsnmr: cuda implementation of xmarginal_nd. */

namespace vrlsnmr::cuda {

namespace impl {

template<typename scalar_t>
__global__ void xmarginal_nd(
  const Grid grid,
  const accessor_t<c10::complex<scalar_t>, 4> kernel_inv,
  const accessor_t<int64_t, 2> ids,
  accessor_t<scalar_t, 3> hist
) {
  const int64_t bs = kernel_inv.size(/*dim=*/0);
  const int64_t nch = kernel_inv.size(/*dim=*/1);
  const int64_t m = kernel_inv.size(/*dim=*/2);

  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  if (i >= m || j >= m)
    return;

  int64_t delta[Grid::max_ndim];
  for (int d = 0; d < grid.ndim; ++d)
    delta[d] = ids[j][d] - ids[i][d];

  for (int64_t bc = blockIdx.z; bc < bs * nch; bc += gridDim.z) {
    const int64_t b = bc / nch;
    const int c = bc % nch;

    const int64_t idx = grid.index(c, delta);
    const auto elem = kernel_inv[b][c][i][j];
    atomicAdd(&hist[b][idx][0], elem.real());
    atomicAdd(&hist[b][idx][1], elem.imag());
  }
}

} /* namespace vrlsnmr::cuda::impl */

/* kernel_inv: (bs, c, m, m), where c=2^(d-1)
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
  const int64_t nch = grid.num_channels();

  const int64_t bs = weights.size(/*dim=*/0);
  const int64_t m = ids.size(/*dim=*/0);

  auto hist = torch::zeros({bs, grid.size}, /*options=*/kernel_inv.options());
  auto hist_real = torch::view_as_real(hist);

  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 threads(32, 8);
  const dim3 blocks(
    (m + threads.x - 1) / threads.x,
    (m + threads.y - 1) / threads.y,
    std::min(bs * nch, max_grid_yz)
  );

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "xmarginal_nd", ([&] {
    impl::xmarginal_nd<scalar_t><<<blocks, threads, 0, stream>>>(
      grid,
      access_as<c10::complex<scalar_t>, 4>(kernel_inv),
      access_as<int64_t, 2>(ids),
      access_as<scalar_t, 3>(hist_real)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }));

  std::vector<int64_t> fft_dims(grid.ndim);
  for (int d = 0; d < grid.ndim; ++d)
    fft_dims[d] = d + 1;

  const auto hist_fft = torch::fft::ifftn(
    /*input=*/hist.view(weights.sizes()),
    /*s=*/c10::nullopt,
    /*dim=*/fft_dims
  );
  const auto sums = torch::real(hist_fft);
  return weights.reciprocal() - sums / (nch * weights.square());
}

} /* namespace vrlsnmr::cuda */
