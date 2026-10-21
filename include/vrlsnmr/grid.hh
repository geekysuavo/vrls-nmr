/* vrlsnmr: sparse multidimensional grids. */

#ifdef __CUDACC__
#define VRLSNMR_HOST __host__
#define VRLSNMR_DEVICE __device__
#else
#define VRLSNMR_HOST
#define VRLSNMR_DEVICE
#endif

namespace vrlsnmr {

/* Grid: multidimensional sampling grid.
 *
 * grids have `d` dimensions and shape `(n_1, ..., n_d)`. grid indices
 * are flattened in row-major order.
 */
struct Grid {
  /* maximum number of indirect grid dimensions. */
  static constexpr int max_ndim = 4;

  /* ndim: number of dimensions.
   * size: number of grid points.
   * shape: per-dimension grid sizes.
   * stride: per-dimension grid strides.
   */
  int ndim;
  int64_t size;
  int64_t shape[max_ndim];
  int64_t stride[max_ndim];

  /* empty constructor.
   */
  Grid() : ndim{0}, size{0}, shape{}, stride{} {}

  /* grid constructor from shape.
   */
  Grid(const std::vector<int64_t>& n) : Grid{} {
    if (n.size() > max_ndim)
      throw std::out_of_range{"grid dimensionality out of supported range"};

    ndim = n.size();
    size = 1;
    for (int dim = ndim - 1; dim >= 0; --dim) {
      shape[dim] = n[dim];
      stride[dim] = size;
      size *= shape[dim];
    }
  }

  /* grid constructor from weights: (bs, n_1, ..., n_d)
   */
  Grid(const torch::Tensor& weights) : Grid{} {
    if (weights.dim() < 2)
      throw std::length_error{"weights have too few dimensions"};

    ndim = int(weights.dim() - 1);
    size = 1;
    for (int dim = ndim - 1; dim >= 0; --dim) {
      shape[dim] = weights.size(/*dim=*/dim + 1);
      stride[dim] = size;
      size *= shape[dim];
    }
  }

  /* number of complex channels.
   */
  VRLSNMR_HOST VRLSNMR_DEVICE
  int64_t num_channels() const {
    return static_cast<int64_t>(1) << (ndim - 1);
  }

  /* whether a zero-based dimension is sign-flipped in a complex channel.
   */
  VRLSNMR_HOST VRLSNMR_DEVICE
  bool flipped(int channel, int dim) const {
    return dim >= 1 && ((channel >> (dim - 1)) & 1);
  }

  /* flat index of a per-dimension index difference, within a given complex channel.
   */
  VRLSNMR_HOST VRLSNMR_DEVICE
  int64_t index(int channel, const int64_t* delta) const {
    int64_t flat = 0;

    for (int dim = 0; dim < ndim; ++dim) {
      const int64_t idx =
        (flipped(channel, dim) ? -delta[dim] : delta[dim]) % shape[dim];

      flat += (idx < 0 ? idx + shape[dim] : idx) * stride[dim];
    }

    return flat;
  }

  /* fft of the reciprocals weights over the grid, conjugated and normalized:
   *
   *  coeffs[delta] = (1 / N) sum_k exp(+2 pi i k.delta / n) / w_k,
   *  where N = prod_k n_k.
   *
   * weights: (bs, n_1, ..., n_d)
   * out => (bs, N)
   */
  torch::Tensor make_coeffs(const torch::Tensor& weights) const {
    std::vector<int64_t> fft_dims(ndim);
    for (int dim = 0; dim < ndim; ++dim)
      fft_dims[dim] = dim + 1;

    auto fft = torch::fft::fftn(
      /*input=*/weights.reciprocal(),
      /*s=*/c10::nullopt,
      /*dim=*/fft_dims
    );
    return (fft.conj() / size).reshape({weights.size(/*dim=*/0), size});
  }

  /* return a grid with half the size as this one.
   */
  Grid half() const {
    std::vector<int64_t> n_half(ndim);
    for (int d = 0; d < ndim; ++d)
      n_half[d] = shape[d] / 2;

    return Grid{n_half};
  }

  /* call a function that deduces its dimensionality template parameter `Dims`
   * based on the type of its first argument.
   */
  template<int Dims = 1, typename Fn>
  void dispatch(Fn&& func) const {
    if constexpr (Dims > max_ndim)
      throw std::out_of_range{"unsupported indirect dimension count"};
    else if (ndim == Dims)
      func(std::integral_constant<int, Dims>{});
    else
      dispatch<Dims + 1>(func);
  }
};

} /* namespace vrlsnmr */
