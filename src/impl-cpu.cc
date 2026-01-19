/* vrlsnmr: custom cpu operator implementations. */

#include <torch/script.h>
#include <torch/extension.h>

template<typename scalar_t, size_t ndim>
using accessor_t = torch::TensorAccessor<scalar_t, ndim>;

template<typename scalar_t, size_t ndim>
accessor_t<scalar_t, ndim> access_as(torch::Tensor x) {
  return x.accessor<scalar_t, ndim>();
}

#include <vrlsnmr/kernel/cpu.hh>
#include <vrlsnmr/xmarginal/cpu.hh>
#include <vrlsnmr/ymarginal/cpu.hh>
#include <vrlsnmr/schedexp/cpu.hh>
#include <vrlsnmr/schedpg/cpu.hh>

TORCH_LIBRARY_IMPL(vrlsnmr, CPU, m) {
  m.impl("kernel", vrlsnmr::cpu::kernel);
  m.impl("xmarginal", vrlsnmr::cpu::xmarginal);
  m.impl("ymarginal", vrlsnmr::cpu::ymarginal);
  m.impl("schedexp", vrlsnmr::cpu::schedexp);
  m.impl("schedpg", vrlsnmr::cpu::schedpg);
}
