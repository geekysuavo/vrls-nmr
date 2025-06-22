/* vrlsnmr: cuda implementation of xmarginal. */

#include <c10/util/MathConstants.h>

namespace vrlsnmr::cuda {

namespace impl {

template<typename scalar_t>
__global__ void xmarginal(
  const accessor_t<c10::complex<scalar_t>, 3> kernel_inv,
  const accessor_t<scalar_t, 2> weights,
  const accessor_t<int64_t, 1> ids,
  accessor_t<scalar_t, 2> out
) {
  const int bs = weights.size(/*dim=*/0);
  const int n = weights.size(/*dim=*/1);
  const int m = ids.size(/*dim=*/0);

  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (b < bs && k < n) {
    const auto theta = 2 * c10::pi<scalar_t> / n;
    const auto wk = weights[b][k];
    scalar_t sum = 0;

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        const auto theta_ij = theta * k * (ids[j] - ids[i]);
        const c10::complex<scalar_t> alpha_ij(cos(theta_ij), sin(theta_ij));

        sum += (kernel_inv[b][i][j] * alpha_ij).real();
      }
    }

    out[b][k] = 1 / wk - sum / (n * wk * wk);
  }
}

} /* namespace vrlsnmr::cuda::impl */

/* kernel_inv: (bs, m, m)
 * weights: (bs, n)
 * ids: (m,)
 * out => (bs, n)
 */
torch::Tensor xmarginal(
  const torch::Tensor& kernel_inv,
  const torch::Tensor& weights,
  const torch::Tensor& ids
) {
  const int64_t bs = weights.size(/*dim=*/0);
  const int64_t n = weights.size(/*dim=*/1);

  auto out = torch::empty({bs, n}, /*options=*/weights.options());

  const dim3 threads(32, 32);
  const dim3 blocks(
    (bs + threads.x - 1) / threads.x,
    (n + threads.y - 1) / threads.y
  );

  AT_DISPATCH_FLOATING_TYPES(
    weights.scalar_type(),
    "xmarginal", ([&] {
    impl::xmarginal<scalar_t><<<blocks, threads>>>(
      access_as<c10::complex<scalar_t>, 3>(kernel_inv),
      access_as<scalar_t, 2>(weights),
      access_as<int64_t, 1>(ids),
      access_as<scalar_t, 2>(out)
    );
  }));

  return out;
}

} /* namespace vrlsnmr::cuda */
