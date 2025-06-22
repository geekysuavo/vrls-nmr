/* vrlsnmr: cuda implementation of kernel. */

namespace vrlsnmr::cuda {

template<typename scalar_t>
__global__ void kernel_forward(
  const scalar_t* tau_ptr,
  const accessor_t<c10::complex<scalar_t>, 1> coeffs,
  const accessor_t<int64_t, 1> ids,
  accessor_t<c10::complex<scalar_t>, 2> out
) {
  const int m = out.size(/*dim=*/0);
  const int n = coeffs.size(/*dim=*/0);

  const int ij = blockIdx.x * blockDim.x + threadIdx.x;

  if (ij < m * m) {
    const int i = ij / m;
    const int j = ij - i * m;

    const scalar_t tau_inv = 1 / *tau_ptr;

    if (i == j) {
      out[i][j] = coeffs[0] + tau_inv;
    }
    else if (i < j) {
      const int k = n + ids[i] - ids[j];
      const auto Kij = coeffs[k];

      out[i][j] = Kij;
      out[j][i] = std::conj(Kij);
    }
  }
}

/* weights: (n,)
 * ids: (m,)
 * tau: ()
 */
torch::Tensor kernel(
  const torch::Tensor& weights,
  const torch::Tensor& ids,
  const torch::Tensor& tau
) {
  using cfloat = c10::complex<float>;

  const int64_t n = weights.size(/*dim=*/0);
  const int64_t m = ids.size(/*dim=*/0);

  const auto coeffs = torch::fft::fft(weights.reciprocal()).conj() / n;

  auto out = torch::empty({m, m}, /*options=*/coeffs.options());

  const int threads = 1024;
  const dim3 blocks((m * m + threads - 1) / threads);

  kernel_forward<float><<<blocks, threads>>>(
    tau.data_ptr<float>(),
    access_as<cfloat, 1>(coeffs),
    access_as<int64_t, 1>(ids),
    access_as<cfloat, 2>(out)
  );

  return out;
}

} /* namespace vrlsnmr::cuda */
