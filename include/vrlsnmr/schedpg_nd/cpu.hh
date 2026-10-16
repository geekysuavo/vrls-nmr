/* vrlsnmr: cpu implementation of schedpg_nd. */

namespace vrlsnmr::cpu {

namespace impl {

/* adapted from <https://github.com/geeksuavo/nusutils> */
template<typename scalar_t>
void schedpg_nd(const std::vector<int64_t>& n, accessor_t<scalar_t, 2> out) {
  /* FIXME: implement vrlsnmr::cpu::impl::schedpg_nd() */
}

} /* namespace vrlsnmr::cpu::impl */

/* n: grid size.
 * out: (m, d)
 */
void schedpg_nd(const std::vector<int64_t>& n, torch::Tensor& out) {
  AT_DISPATCH_INTEGRAL_TYPES(
    out.scalar_type(),
    "schedpg_nd", ([&] {
      impl::schedpg_nd<scalar_t>(n, access_as<scalar_t, 2>(out));
  }));
}

} /* namespace vrlsnmr::cpu */
