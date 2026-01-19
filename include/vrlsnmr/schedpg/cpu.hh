/* vrlsnmr: cpu implementation of schedpg. */

namespace vrlsnmr::cpu {

namespace impl {

/* adapted from <https://gwagner.hms.harvard.edu/poisson-gap-sampling> */
template<typename scalar_t>
void schedpg(int64_t n, accessor_t<scalar_t, 1> out) {
  const auto uniform = [] () -> double {
    return torch::rand({}, torch::kFloat64).item<double>();
  };

  const auto poisson = [&] (double rate) -> scalar_t {
    const double exp_rate = std::exp(-rate);
    scalar_t draw = 0;
    double prob = 1.0;

    do {
      prob *= uniform();
      draw++;
    } while (prob >= exp_rate);

    return draw - 1;
  };

  const int64_t m = out.size(/*dim=*/0);
  int64_t num_samples;
  scalar_t index;

  const double base_rate = double(n) / double(m) - 1.0;
  double weight = 2.0;
  
  do {
    num_samples = 0;
    index = 0;

    while (index < n) {
      if (num_samples < m)
        out[num_samples] = index;

      num_samples++;
      index++;

      const double theta = c10::pi<double> * double(index) / double(n + 1);
      const double rate = base_rate * weight * std::sin(0.5 * theta);
      index += poisson(rate);
    }

    if (num_samples > m) weight *= 1.02;
    if (num_samples < m) weight /= 1.02;
  } while (num_samples != m);
}

} /* namespace vrlsnmr::cpu::impl */

/* n: grid size.
 * out: (m,)
 */
void schedpg(int64_t n, torch::Tensor& out) {
  AT_DISPATCH_INTEGRAL_TYPES(
    out.scalar_type(),
    "schedpg", ([&] {
      impl::schedpg<scalar_t>(n, access_as<scalar_t, 1>(out));
  }));
}

} /* namespace vrlsnmr::cpu */
