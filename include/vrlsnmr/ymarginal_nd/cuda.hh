/* vrlsnmr: cuda implementation of ymarginal_nd. */

namespace vrlsnmr::cuda {

namespace impl {

struct LaunchConfig {
  dim3 threads;
  dim3 blocks;
  int64_t splits;
};

template<typename scalar_t, int Dims, int RowsPerPass = 4, int BinsPerThread = 4>
__global__ void ymarginal_nd(
  const Grid grid,
  const Grid grid_out,
  const accessor_t<c10::complex<scalar_t>, 4> kernel_inv,
  const accessor_t<c10::complex<scalar_t>, 2> coeffs,
  const accessor_t<int64_t, 2> ids,
  accessor_t<scalar_t, 3> out
) {
  // FIXME -- implement vrlsnmr::cuda::impl::ymarginal_nd()
}

template<int RowsPerPass = 4, int BinsPerThread = 4, int Threads = 128>
LaunchConfig ymarginal_nd_launch(int64_t kmax, int64_t bs, int64_t m) {
  const int64_t threads = static_cast<int64_t>(Threads);

  const int64_t k_threads = threads * BinsPerThread;
  const int64_t k_blocks = (kmax + k_threads - 1) / k_threads;

  const int64_t b_blocks = std::min(bs, max_grid_yz);

  const auto props = at::cuda::getCurrentDeviceProperties();
  const int64_t target = 4 * props->multiProcessorCount;

  const int64_t row_blocks = std::max<int64_t>((m + RowsPerPass - 1) / RowsPerPass, 1);
  const int64_t splits = std::clamp<int64_t>(
    (target + k_blocks * b_blocks - 1) / (k_blocks * b_blocks),
    /*lo=*/1, /*hi=*/row_blocks
  );

  return {dim3(threads), dim3(k_blocks, splits, b_blocks), splits};
}

torch::Tensor ymarginal_nd_finish(
  const torch::Tensor& coeffs,
  const torch::Tensor& partial,
  const Grid& grid_out
) {
  const auto c0 = torch::real(coeffs.select(/*dim=*/1, /*index=*/0))
                        .unsqueeze(/*dim=*/1);

  const auto out = c0 - partial.sum(/*dim=*/0) / grid_out.num_channels();

  std::vector<int64_t> shape{out.size(/*dim=*/0)};
  for (int d = 0; d < grid_out.ndim; ++d)
    shape.push_back(grid_out.shape[d]);

  return out.view(shape);
}

} /* namespace vrlsnmr::cuda::impl */

/* kernel_inv: (b, c, m, m), where c=2^(d-1)
 * weights: (b, n_1, ..., n_d)
 * ids: (m, d)
 * out => (b, n_1, ..., n_d) or (b, n_1 / 2, ..., n_d / 2)
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
  const int64_t m = ids.size(/*dim=*/0);

  const auto config = impl::ymarginal_nd_launch(grid_out.size, bs, m);
  auto out = torch::empty(
    {config.splits, bs, grid_out.size},
    /*options=*/weights.options()
  );

  const auto coeffs = grid.make_coeffs(weights);
  auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "ymarginal_nd", ([&] {
    grid.dispatch([&] (auto dims_tag) {
      constexpr int Dims = decltype(dims_tag)::value;
      impl::ymarginal_nd<scalar_t, Dims><<<config.blocks, config.threads, 0, stream>>>(
        grid,
        grid_out,
        access_as<c10::complex<scalar_t>, 4>(kernel_inv),
        access_as<c10::complex<scalar_t>, 2>(coeffs),
        access_as<int64_t, 2>(ids),
        access_as<scalar_t, 3>(out)
      );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }));

  return impl::ymarginal_nd_finish(coeffs, out, grid_out);
}

} /* namespace vrlsnmr::cuda */
