/* vrlsnmr: cpu implementation of schedexp. */

namespace vrlsnmr::cpu {

namespace impl {

template<typename scalar_t>
void schedexp(double rate, int64_t n, accessor_t<scalar_t, 1> out) {
  const auto uniform = [] () -> double {
    return torch::rand({}, torch::kFloat64).item<double>();
  };

  const auto rejection = [&] () -> scalar_t {
    scalar_t draw = 0;
    do {
      draw = uniform() * n;
    } while (uniform() > std::exp(-rate * draw));

    return draw;
  };

  const int64_t m = out.size(/*dim=*/0);
  std::unordered_set<scalar_t> samples;
  int64_t sample_index = 0;
  do {
    const scalar_t s = rejection();
    if (samples.find(s) == samples.end()) {
      samples.insert(s);
      out[sample_index] = s;
      sample_index++;
    }
  } while (sample_index < m);

  std::sort(/*begin=*/out.data(), /*end=*/out.data() + m);
}

} /* namespace vrlsnmr::cpu::impl */

/* rate: exponential pdf parameter.
 * n: grid size.
 * out: (m,)
 */
void schedexp(double rate, int64_t n, torch::Tensor& out) {
  AT_DISPATCH_INTEGRAL_TYPES(
    out.scalar_type(),
    "schedexp", ([&] {
      impl::schedexp<scalar_t>(rate, n, access_as<scalar_t, 1>(out));
  }));
}

} /* namespace vrlsnmr::cpu */
