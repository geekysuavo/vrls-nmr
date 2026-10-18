/* vrlsnmr: cpu implementation of schedpg_nd. */

namespace vrlsnmr::cpu {

namespace impl {

/* adapted from <https://github.com/geeksuavo/nusutils> */
template<typename scalar_t>
class PoissonGap {
public:
  /* constructor taking grid size.
   */
  PoissonGap(const std::vector<int64_t>& n) {
    grid = Grid{n};

    { /* only hold the lock in this scope. */
      auto gen = at::detail::getDefaultCPUGenerator();
      std::lock_guard<std::mutex> lock(gen.mutex());
      rng.seed(gen.get<at::CPUGeneratorImpl>()->random64());
    }

    free_dims.resize(grid.ndim);
    origin.resize(grid.ndim);

    grid_sum = std::accumulate(n.begin(), n.end(), static_cast<int64_t>(0));
  }

  /* return the total number of grid points.
   */
  int64_t size() const {
    return grid.size;
  }

  /* sample a multidimensional grid.
   */
  int64_t sample(double scale, accessor_t<scalar_t, 2>& out) {
    std::fill(free_dims.begin(), free_dims.end(), true);
    std::fill(origin.begin(), origin.end(), 0);
    samples.clear();

    sample_grid(scale, grid.ndim);

    const int64_t num_samples = samples.size();
    if (num_samples == out.size(/*dim=*/0)) {
      int64_t row = 0;
      for (const auto sample : samples)
        store(sample, row++, out);
    }

    return num_samples;
  }

private:
  Grid grid;
  std::mt19937 rng;
  std::vector<bool> free_dims;
  std::vector<int64_t> origin;
  std::unordered_set<int64_t> samples;
  int64_t grid_sum;

  /* draw a single gap from a poisson distribution.
   */
  int64_t draw(double rate) {
    return std::poisson_distribution<scalar_t>(rate)(rng) + 1;
  }

  /* unpack an index into an output tensor accessor row.
   */
  void store(int64_t index, int64_t row, accessor_t<scalar_t, 2>& out) {
    if (row >= out.size(/*dim=*/0))
      return;

    int64_t reduced = index;
    for (int dim = 0; dim < grid.ndim; ++dim) {
      const int64_t dim_index = reduced % grid.shape[dim];
      reduced = (reduced - dim_index) / grid.shape[dim];

      out[row][dim] = static_cast<scalar_t>(dim_index);
    }
  }

  /* sample along a single dimension at a single grid origin.
   */
  void sample_dim(
    double scale,
    int64_t dim,
    int64_t base,
    int64_t origin_sum
  ) {
    static constexpr double pi = c10::pi<double>;
    int64_t idx = 0;
    while (true) {
      const double theta = 0.5 * pi * double(idx + origin_sum) / double(grid_sum);
      idx += draw(scale * theta);
      if (idx <= grid.shape[dim]) {
        samples.insert(base + (idx - 1) * grid.stride[dim]);
      }
      else
        return;
    }
  }

  /* sample along multiple grid dimensions recursively.
   */
  void sample_grid(double scale, int num_free_dims) {
    if (num_free_dims == 1) {
      const auto dir = std::distance(
        free_dims.cbegin(),
        std::find(free_dims.cbegin(), free_dims.cend(), true)
      );

      int64_t base = 0;
      int64_t origin_sum = 0;
      for (int dim = 0; dim < grid.ndim; ++dim) {
        base += origin[dim] * grid.stride[dim];
        origin_sum += origin[dim];
      }

      sample_dim(scale, dir, base, origin_sum);
    }
    else {
      for (int64_t offset = 0;; ++offset) {
        bool recursed = false;
        for (int dim = 0; dim < grid.ndim; ++dim) {
          if (!free_dims[dim] || offset >= grid.shape[dim])
            continue;

          origin[dim] = offset;
          free_dims[dim] = false;

          sample_grid(scale, num_free_dims - 1);

          origin[dim] = 0;
          free_dims[dim] = true;

          recursed = true;
        }

        if (!recursed)
          break;
      }
    }
  }
};

template<typename scalar_t>
void schedpg_nd(const std::vector<int64_t>& n, accessor_t<scalar_t, 2> out) {
  PoissonGap<scalar_t> pg{n};
  const int64_t m = out.size(/*dim=*/0);
  double scale = double(pg.size()) / double(m) - 1.0;

  for (int iter = 0; iter < 1000; ++iter) {
    const int64_t num_samples = pg.sample(scale, out);
    if (num_samples == m)
      break;
    else
      scale *= 1.0 + 0.5 * double(num_samples - m) / double(m);
  }
}

} /* namespace vrlsnmr::cpu::impl */

/* n: grid size.
 * out: (m, d)
 */
void schedpg_nd(const std::vector<int64_t>& n, torch::Tensor& out) {
  AT_DISPATCH_INTEGRAL_TYPES(
    out.scalar_type(),
    "schedpg_nd", ([&] {
      impl::schedpg_nd<scalar_t>(n, access_as<scalar_t, 2>(out));
  }));
}

} /* namespace vrlsnmr::cpu */
