/* vrlsnmr: custom cuda operator implementations. */

#include <torch/script.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MathConstants.h>

/* maximum size of gridDim.y and gridDim.z */
constexpr int64_t max_grid_yz = 65535;

template<typename scalar_t, size_t ndim>
using accessor_t =
  torch::PackedTensorAccessor32<scalar_t, ndim, torch::RestrictPtrTraits>;

template<typename scalar_t, size_t ndim>
accessor_t<scalar_t, ndim> access_as(torch::Tensor x) {
  return x.packed_accessor32<scalar_t, ndim, torch::RestrictPtrTraits>();
}

#include <vrlsnmr/grid.hh>
#include <vrlsnmr/kernel/cuda.hh>
#include <vrlsnmr/kernel_nd/cuda.hh>
#include <vrlsnmr/xmarginal/cuda.hh>
#include <vrlsnmr/xmarginal_nd/cuda.hh>
#include <vrlsnmr/ymarginal/cuda.hh>
#include <vrlsnmr/ymarginal_nd/cuda.hh>

TORCH_LIBRARY_IMPL(vrlsnmr, CUDA, m) {
  m.impl("kernel", vrlsnmr::cuda::kernel);
  m.impl("xmarginal", vrlsnmr::cuda::xmarginal);
  m.impl("ymarginal", vrlsnmr::cuda::ymarginal);

  m.impl("kernel_nd", vrlsnmr::cuda::kernel_nd);
  m.impl("xmarginal_nd", vrlsnmr::cuda::xmarginal_nd);
  m.impl("ymarginal_nd", vrlsnmr::cuda::ymarginal_nd);
}
