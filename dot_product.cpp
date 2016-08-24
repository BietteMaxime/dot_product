#include <iostream>
#include <type_traits>
#include <cassert>
#include <cstdlib>
#include <chrono>

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif // __SSE4_1__

#if defined(__AVX__)
#include <immintrin.h>
#endif // __AVX__

using clock_ = std::chrono::high_resolution_clock;

template <typename T>
auto to_ns(T const& val)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(val);
}


union UnitVec
{
#if defined(__AVX__)
  using intrinsic = __m256;
#elif defined(__SSE4_1__)
  using intrinsic = __m128;
#endif
  static constexpr size_t size = sizeof(intrinsic)/sizeof(float);

  float const& operator[](size_t idx) const
  {
    assert(idx < size);
    return f[idx];
  }

  float& operator[](size_t idx)
  {
    assert(idx < size);
    return f[idx];
  }

  float f[size];
  intrinsic i;
} __attribute__((aligned(16)));

float dot_product(UnitVec const& a, UnitVec const& b)
{
  UnitVec r;
#if defined(__AVX__)
  r.i = _mm256_dp_ps(a.i, b.i, 0xf1);
  return r[0] + r[4];
#else
  r.i = _mm_dp_ps(a.i, b.i, 0xf1);
  return r[0];
#endif
}

float dot_product_trivial(UnitVec const& a, UnitVec const& b)
{
  float r = 0.0f;
  for (size_t i = 0; i < UnitVec::size; ++i)
  {
    r += a[i] * b[i];
  }
  return r;
}


template <size_t N>
struct StaticVec
{
  static constexpr size_t size = N;
  static constexpr size_t nb_unitvecs = N / UnitVec::size;

  float const& operator[](size_t idx) const
  {
    assert(idx < N);
    const auto i = std::div(static_cast<long long int>(idx),
                            static_cast<long long int>(UnitVec::size));
    // std::cout << __PRETTY_FUNCTION__
    //           << " idx " << idx << " n " << nb_unitvecs
    //           << " quot " << i.quot << " rem " << i.rem
    //           << std::endl;
    return data[i.quot][i.rem];
  }

  float& operator[](size_t idx)
  {
    assert(idx < N);
    const auto i = std::div(static_cast<long long int>(idx),
                            static_cast<long long int>(UnitVec::size));
    // std::cout << __PRETTY_FUNCTION__
    //           << " idx " << idx << " n " << nb_unitvecs
    //           << " quot " << i.quot << " rem " << i.rem
    //           << std::endl;
    return data[i.quot][i.rem];
  }

  UnitVec data[nb_unitvecs];
};

template <size_t N>
float dot_product_trivial(StaticVec<N> const& l,
                          StaticVec<N> const& r)
{
  float ret = {0};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    ret += dot_product_trivial(l.data[i], r.data[i]);
  }
  return ret;
}

template <size_t N>
float dot_product(StaticVec<N> const& l, StaticVec<N> const& r)
{
  float ret = {0};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    ret += dot_product(l.data[i], r.data[i]);
  }
  return ret;
}


int main()
{
  UnitVec a, b/*, r = {0}*/;

  for (size_t i = 0; i < UnitVec::size; ++i)
  {
    a.f[i] = b.f[UnitVec::size - i - 1] = i;
  }

  for (size_t i = 0; i < UnitVec::size; ++i)
  {
      std::cout << i
              << " a " << a[i]
              << " b " << b[i]
              << std::endl;
  }


  std::cout << "UnitVec size " << UnitVec::size
            << " " << dot_product(a, b) << " " << dot_product_trivial(a, b)
            << std::endl;

  constexpr size_t N = 1 << 13;
  StaticVec<N> u, v;
  for (size_t i = 0; i < StaticVec<N>::size; ++i)
  {
    u[i] = v[StaticVec<N>::size - i - 1] = i;
  }

  for (size_t i = 0; i < StaticVec<N>::size; ++i)
  {
      std::cout << i
              << " u " << u[i]
              << " v " << v[i]
              << std::endl;
  }

  const auto start_res1 = clock_::now();
  const auto res1 = dot_product_trivial(u, v);
  const auto elapsed_res1 = to_ns(clock_::now() - start_res1);

  std::cout << "StaticVec trivial size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res1
            << " elapsed " << elapsed_res1.count() << "ns"
            << std::endl;

  const auto start_res2 = clock_::now();
  const auto res2 = dot_product(u, v);
  const auto elapsed_res2 = to_ns(clock_::now() - start_res2);

  std::cout << "StaticVec fast size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res2
            << " elapsed " << elapsed_res2.count() << "ns"
            << std::endl;

  return 0;
}
