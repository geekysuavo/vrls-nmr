/* vrlsnmr: custom cuda operator implementations. */

#include <torch/script.h>
#include <torch/extension.h>

template<typename scalar_t, size_t ndim>
using accessor_t =
  torch::PackedTensorAccessor32<scalar_t, ndim, torch::RestrictPtrTraits>;

template<typename scalar_t, size_t ndim>
accessor_t<scalar_t, ndim> access_as(torch::Tensor x) {
  return x.packed_accessor32<scalar_t, ndim, torch::RestrictPtrTraits>();
}

#include <vrlsnmr/kernel/cuda.hh>
#include <vrlsnmr/xmarginal/cuda.hh>
#include <vrlsnmr/ymarginal/cuda.hh>

TORCH_LIBRARY_IMPL(vrlsnmr, CUDA, m) {
  m.impl("kernel", vrlsnmr::cuda::kernel);
  m.impl("xmarginal", vrlsnmr::cuda::xmarginal);
  m.impl("ymarginal", vrlsnmr::cuda::ymarginal);
}
