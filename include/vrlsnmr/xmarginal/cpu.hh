/* vrlsnmr: cpu implementation of xmarginal. */

#include <c10/util/MathConstants.h>

namespace vrlsnmr::cpu {

/* kernel_inv: (m, m)
 * weights: (n,)
 * ids: (m,)
 */
torch::Tensor xmarginal(
  const torch::Tensor& kernel_inv,
  const torch::Tensor& weights,
  const torch::Tensor& ids
) {
  using cfloat = c10::complex<float>;

  const int64_t n = weights.size(/*dim=*/0);
  const int64_t m = ids.size(/*dim=*/0);

  auto out = torch::empty({n}, /*options=*/weights.options());
  const auto kernel_inv_accessor = kernel_inv.accessor<cfloat, 2>();
  const auto weights_accessor = weights.accessor<float, 1>();
  const auto ids_accessor = ids.accessor<int64_t, 1>();
  auto out_accessor = out.accessor<float, 1>();

  const auto theta = 2.0f * c10::pi<float> / n;
  for (int64_t k = 0; k < n; ++k) {
    const auto wk = weights_accessor[k];
    float sum = 0.0f;

    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < m; ++j) {
        const auto theta_ij = theta * k * (ids_accessor[j] - ids_accessor[i]);
        const cfloat alpha_ij{std::cos(theta_ij), std::sin(theta_ij)};

        sum += (kernel_inv_accessor[i][j] * alpha_ij).real();
      }
    }

    out_accessor[k] = 1 / wk - sum / (n * wk * wk);
  }

  return out;
}

} /* namespace vrlsnmr::cpu */
