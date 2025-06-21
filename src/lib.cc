/* vrlsnmr: operator function signatures. */

#include <torch/script.h>

TORCH_LIBRARY(vrlsnmr, m) {
  m.def("kernel(Tensor weights, Tensor ids, Tensor tau) -> Tensor");
  m.def("xmarginal(Tensor kernel_inv, Tensor weights, Tensor ids) -> Tensor");
  m.def("ymarginal(Tensor kernel_inv, Tensor weights, Tensor ids) -> Tensor");
}
