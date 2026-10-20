/* vrlsnmr: cuda implementation of kernel_nd. */

namespace vrlsnmr::cuda {

namespace impl {

template<typename scalar_t>
__global__ void kernel_nd(
  const Grid grid,
  const accessor_t<c10::complex<scalar_t>, 2> coeffs,
  const accessor_t<scalar_t, 1> tau,
  const accessor_t<int64_t, 2> ids,
  accessor_t<c10::complex<scalar_t>, 4> out
) {
  const int64_t bs = out.size(/*dim=*/0);
  const int64_t nch = out.size(/*dim=*/1);
  const int64_t m = out.size(/*dim=*/2);

  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  if (i >= m || j >= m)
    return;

  int64_t delta[Grid::max_ndim];
  for (int d = 0; d < grid.ndim; ++d)
    delta[d] = (i <= j) ? ids[i][d] - ids[j][d] : ids[j][d] - ids[i][d];

  for (int64_t bc = blockIdx.z; bc < bs * nch; bc += gridDim.z) {
    c10::complex<scalar_t> elem;
    const int64_t b = bc / nch;
    const int c = bc % nch;

    if (i == j) {
      elem = c10::complex<scalar_t>{coeffs[b][0].real() + 1 / tau[b], 0};
    }
    else if (i < j) {
      const int64_t idx = grid.index(c, delta);
      elem = coeffs[b][idx];
    }
    else {
      const int64_t idx = grid.index(c, delta);
      elem = std::conj(coeffs[b][idx]);
    }

    out[b][c][i][j] = elem;
  }
}

} /* namespace vrlsnmr::cuda::impl */

/* weights: (bs, n_1, ..., n_d)
 * tau: (bs,)
 * ids: (m, d)
 * out => (bs, c, m, m), where c=2^(d-1)
 */
torch::Tensor kernel_nd(
  const torch::Tensor& weights,
  const torch::Tensor& tau,
  const torch::Tensor& ids
) {
  const Grid grid{weights};
  const auto coeffs = grid.make_coeffs(weights);

  const int64_t bs = weights.size(/*dim=*/0);
  const int64_t nch = grid.num_channels();
  const int64_t m = ids.size(/*dim=*/0);

  auto out = torch::empty({bs, nch, m, m}, /*options=*/coeffs.options());

  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 threads(32, 8);
  const dim3 blocks(
    (m + threads.x - 1) / threads.x,
    (m + threads.y - 1) / threads.y,
    std::min(bs * nch, max_grid_yz)
  );

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "kernel_nd", ([&] {
    impl::kernel_nd<scalar_t><<<blocks, threads, 0, stream>>>(
      grid,
      access_as<c10::complex<scalar_t>, 2>(coeffs),
      access_as<scalar_t, 1>(tau),
      access_as<int64_t, 2>(ids),
      access_as<c10::complex<scalar_t>, 4>(out)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }));

  return out;
}

} /* namespace vrlsnmr::cuda */
