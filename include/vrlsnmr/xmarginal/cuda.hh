/* vrlsnmr: cuda implementation of xmarginal. */

#include <c10/util/MathConstants.h>

namespace vrlsnmr::cuda {

template<typename scalar_t>
__global__ void xmarginal_forward(
  const accessor_t<c10::complex<scalar_t>, 2> kernel_inv,
  const accessor_t<scalar_t, 1> weights,
  const accessor_t<int64_t, 1> ids,
  accessor_t<scalar_t, 1> out
) {
  const int m = ids.size(/*dim=*/0);
  const int n = out.size(/*dim=*/0);

  const int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < n) {
    const auto theta = 2.0f * c10::pi<scalar_t> / n;
    const auto wk = weights[k];
    scalar_t sum = 0.0f;

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        const auto theta_ij = theta * k * (ids[j] - ids[i]);
        const c10::complex<scalar_t> alpha_ij(cos(theta_ij), sin(theta_ij));

        sum += (kernel_inv[i][j] * alpha_ij).real();
      }
    }

    out[k] = 1 / wk - sum / (n * wk * wk);
  }
}

/* kernel_inv: (m, m)
 * weights: (n,)
 * ids: (m,)
 */
torch::Tensor xmarginal(
  torch::Tensor kernel_inv,
  torch::Tensor weights,
  torch::Tensor ids
) {
  using cfloat = c10::complex<float>;

  const int64_t n = weights.size(/*dim=*/0);

  auto out = torch::empty({n}, /*options=*/weights.options());

  const int threads = 1024;
  const dim3 blocks((n + threads - 1) / threads);

  xmarginal_forward<float><<<blocks, threads>>>(
    access_as<cfloat, 2>(kernel_inv),
    access_as<float, 1>(weights),
    access_as<int64_t, 1>(ids),
    access_as<float, 1>(out)
  );

  return out;
}

} /* namespace vrlsnmr::cuda */
