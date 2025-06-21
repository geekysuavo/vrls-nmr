/* vrlsnmr: custom cpu operator implementations. */

#include <torch/script.h>
#include <torch/extension.h>

#include <vrlsnmr/kernel/cpu.hh>
#include <vrlsnmr/xmarginal/cpu.hh>
#include <vrlsnmr/ymarginal/cpu.hh>

TORCH_LIBRARY_IMPL(vrlsnmr, CPU, m) {
  m.impl("kernel", vrlsnmr::cpu::kernel);
  m.impl("xmarginal", vrlsnmr::cpu::xmarginal);
  m.impl("ymarginal", vrlsnmr::cpu::ymarginal);
}
