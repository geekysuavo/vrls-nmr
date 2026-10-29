/* vrlsnmr: cuda implementation of ymarginal_nd. */

namespace vrlsnmr::cuda {

namespace impl {

/* kernel launch configuration.
 */
struct LaunchConfig {
  dim3 threads;
  dim3 blocks;
  int64_t splits;
};

/* tile of bins owned by a given thread.
 */
template<int Dims, int BinsPerThread>
struct Tile {
  int64_t flat_ids[BinsPerThread];
  int ids[BinsPerThread][Dims];
  bool valid[BinsPerThread];

  /* tile constructor from an output grid.
   */
  __device__ __forceinline__ Tile(const Grid& grid_out) {
    const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x * BinsPerThread;

    #pragma unroll
    for (int s = 0; s < BinsPerThread; ++s) {
      flat_ids[s] = first + static_cast<int64_t>(s) * blockDim.x + threadIdx.x;

      valid[s] = flat_ids[s] < grid_out.size;
      flat_ids[s] = valid[s] ? flat_ids[s] : grid_out.size - 1;

      #pragma unroll
      for (int d = 0; d < Dims; ++d)
        ids[s][d] = static_cast<int>(
          (flat_ids[s] / grid_out.stride[d]) % grid_out.shape[d]
        );
    }
  }

  /* flatten a grid index difference under sign-flips of a given channel.
   */
  __device__ __forceinline__ int64_t flatten(
    const Grid& grid,
    const int* index,
    int s, /* slot */
    int c  /* channel */
  ) const {
    int64_t flat_index = 0;

    #pragma unroll
    for (int d = 0; d < Dims; ++d) {
      int delta = index[d] - ids[s][d];
      delta = grid.flipped(c, d) ? -delta : delta;
      delta = delta < 0 ? delta + static_cast<int>(grid.shape[d]) : delta;
      flat_index += static_cast<int64_t>(delta) * grid.stride[d];
    }

    return flat_index;
  }
};

/* load a grid index from a tensor accessor.
 */
template<int Dims>
__device__ __forceinline__ void ymarginal_nd_load_index(
  const accessor_t<int64_t, 2>& ids,
  const int64_t i,
  int* index
) {
  #pragma unroll
  for (int d = 0; d < Dims; ++d)
    index[d] = static_cast<int>(ids[i][d]);
}

/* out[r][s] += sum_{j >= i} kernel_inv[i][j] coeffs[t_j - k_s]
 */
template<typename scalar_t, int Dims, int RowsPerPass, int BinsPerThread>
__device__ __forceinline__ void ymarginal_nd_accumulate(
  const Grid& grid,
  const accessor_t<c10::complex<scalar_t>, 4>& kernel_inv,
  const accessor_t<c10::complex<scalar_t>, 2>& coeffs,
  const accessor_t<int64_t, 2>& ids,
  const Tile<Dims, BinsPerThread>& bins,
  int64_t start,
  int64_t b,
  int c,
  c10::complex<scalar_t> (&out)[RowsPerPass][BinsPerThread]
) {
  const int64_t m = ids.size(/*dim=*/0);

  for (int64_t j = start; j < m; ++j) {
    int index[Dims];
    ymarginal_nd_load_index<Dims>(ids, j, index);

    c10::complex<scalar_t> beta_j[BinsPerThread];

    #pragma unroll
    for (int s = 0; s < BinsPerThread; ++s) {
      const int64_t idx_j = bins.flatten(grid, index, s, c);
      beta_j[s] = coeffs[b][idx_j];
    }

    #pragma unroll
    for (int r = 0; r < RowsPerPass; ++r) {
      const int64_t i = start + r;
      if (i < m && j >= i) {
        const scalar_t weight = (i == j) ? scalar_t(1) : scalar_t(2);
        const c10::complex<scalar_t> kij = weight * kernel_inv[b][c][i][j];

        #pragma unroll
        for (int s = 0; s < BinsPerThread; ++s)
          out[r][s] += kij * beta_j[s];
      }
    }
  }
}

/* out[s] += Re(conj(coeffs[t_i - k_s]) acc[r][s])
 */
template<typename scalar_t, int Dims, int RowsPerPass, int BinsPerThread>
__device__ __forceinline__ void ymarginal_nd_contract(
  const Grid& grid,
  const accessor_t<c10::complex<scalar_t>, 2>& coeffs,
  const accessor_t<int64_t, 2>& ids,
  const Tile<Dims, BinsPerThread>& bins,
  int64_t start,
  int64_t b,
  int c,
  const c10::complex<scalar_t> (&acc)[RowsPerPass][BinsPerThread],
  scalar_t (&out)[BinsPerThread]
) {
  const int64_t m = ids.size(/*dim=*/0);

  #pragma unroll
  for (int r = 0; r < RowsPerPass; ++r) {
    if (start + r < m) {
      int index[Dims];
      ymarginal_nd_load_index<Dims>(ids, start + r, index);

      #pragma unroll
      for (int s = 0; s < BinsPerThread; ++s) {
        const int64_t idx_i = bins.flatten(grid, index, s, c);
        const auto beta_i = std::conj(coeffs[b][idx_i]);

        out[s] += (beta_i * acc[r][s]).real();
      }
    }
  }
}

/* register/row tiled implementation.
 */
template<typename scalar_t, int Dims, int RowsPerPass = 4, int BinsPerThread = 4>
__global__ void ymarginal_nd(
  const Grid grid,
  const Grid grid_out,
  const accessor_t<c10::complex<scalar_t>, 4> kernel_inv,
  const accessor_t<c10::complex<scalar_t>, 2> coeffs,
  const accessor_t<int64_t, 2> ids,
  accessor_t<scalar_t, 3> out
) {
  const int64_t bs = kernel_inv.size(/*dim=*/0);
  const int64_t nch = kernel_inv.size(/*dim=*/1);
  const int64_t m = kernel_inv.size(/*dim=*/2);

  const Tile<Dims, BinsPerThread> bins{grid_out};

  for (int64_t b = blockIdx.z; b < bs; b += gridDim.z) {
    scalar_t sums[BinsPerThread];

    #pragma unroll
    for (int s = 0; s < BinsPerThread; ++s)
      sums[s] = 0;

    for (int c = 0; c < nch; ++c) {
      for (
        int64_t start = static_cast<int64_t>(blockIdx.y) * RowsPerPass;
        start < m;
        start += static_cast<int64_t>(gridDim.y) * RowsPerPass
      ) {
        c10::complex<scalar_t> acc[RowsPerPass][BinsPerThread];

        #pragma unroll
        for (int r = 0; r < RowsPerPass; ++r) {
          #pragma unroll
          for (int s = 0; s < BinsPerThread; ++s)
            acc[r][s] = c10::complex<scalar_t>(0, 0);
        }

        ymarginal_nd_accumulate(grid, kernel_inv, coeffs, ids, bins, start, b, c, acc);
        ymarginal_nd_contract(grid, coeffs, ids, bins, start, b, c, acc, sums);
      }
    }

    #pragma unroll
    for (int s = 0; s < BinsPerThread; ++s) {
      if (bins.valid[s])
        out[blockIdx.y][b][bins.flat_ids[s]] = sums[s];
    }
  }
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
