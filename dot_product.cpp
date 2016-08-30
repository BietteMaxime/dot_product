#include <iostream>
#include <type_traits>
#include <cassert>
#include <cstdlib>
#include <chrono>

#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#elif defined(__SSE3__)
#include <pmmintrin.h>
#endif

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
#elif defined(__SSE__)
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

  UnitVec& operator+=(UnitVec const& rhs)
  {
#if defined(__AVX__)
    i = _mm256_add_ps(i, rhs.i);
#elif defined(__SSE__)
    i = _mm_add_ps(i, rhs.i);
#endif
    return *this;
  }

  UnitVec& operator*=(UnitVec const& rhs)
  {
#if defined(__AVX__)
    i = _mm256_mul_ps(i, rhs.i);
#elif defined(__SSE__)
    i = _mm_mul_ps(i, rhs.i);
#endif
    return *this;
  }

  float hadd() const
  {
#if defined(__AVX__)
    UnitVec ret;
    ret.i = _mm256_hadd_ps(this->i, this->i);
    ret.i = _mm256_hadd_ps(ret.i, ret.i);
    ret.i = _mm256_hadd_ps(ret.i, ret.i);
    return ret.f[0];
#elif defined(__SSE3__)
    UnitVec ret;
    ret.i = _mm_hadd_ps(this->i, this->i);
    ret.i = _mm_hadd_ps(ret.i, ret.i);
    return ret.f[0];
#endif
  }

  float f[size];
  intrinsic i;
} __attribute__((aligned(16)));

UnitVec operator*(UnitVec const& lhs, UnitVec const& rhs)
{
  UnitVec ret{{0}};
#if defined(__AVX__)
  ret.i = _mm256_mul_ps(lhs.i, rhs.i);
#else
  ret.i = _mm_mul_ps(lhs.i, rhs.i);
#endif
  return ret;
}

UnitVec operator+(UnitVec const& lhs, UnitVec const& rhs)
{
  UnitVec ret{{0}};
#if defined(__AVX__)
  ret.i = _mm256_add_ps(lhs.i, rhs.i);
#else
  ret.i = _mm_add_ps(lhs.i, rhs.i);
#endif
  return ret;
}

float dot_product__01(UnitVec const& a, UnitVec const& b)
{
  UnitVec r;
#if defined(__AVX__)
  r.i = _mm256_dp_ps(a.i, b.i, 0xf1);
  return r[0] + r[4];
#elif defined(__SSE4_1__)
  r.i = _mm_dp_ps(a.i, b.i, 0xf1);
  return r[0];
#elif defined(__SSE3__)
  r.i = _mm_mul_ps(a.i, b.i);
  r.i = _mm_hadd_ps(r.i, r.i);
  r.i = _mm_hadd_ps(r.i, r.i);
  return r[0];
#endif
}

template <size_t N>
struct StaticVec
{
  static_assert(N % UnitVec::size == 0, "N not a multiple of UnitVec::size");
  static constexpr size_t size = N;
  static constexpr size_t nb_unitvecs = N / UnitVec::size;

  float const& operator[](size_t idx) const
  {
    assert(idx < N);
    const auto i = std::div(static_cast<long long int>(idx),
                            static_cast<long long int>(UnitVec::size));
    return data[i.quot][i.rem];
  }

  float& operator[](size_t idx)
  {
    assert(idx < N);
    const auto i = std::div(static_cast<long long int>(idx),
                            static_cast<long long int>(UnitVec::size));
    return data[i.quot][i.rem];
  }

  StaticVec<N>& operator*=(StaticVec<N> const& rhs)
  {
    for (size_t i = 0; i < nb_unitvecs; ++i)
    {
      data[i] *= rhs.data[i];
    }
    return *this;
  }

  UnitVec data[nb_unitvecs];
};

template <size_t N>
StaticVec<N> operator+(StaticVec<N> const& lhs, StaticVec<N> const& rhs)
{
  StaticVec<N> ret{0};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    ret.data[i] = lhs.data[i] + rhs.data[i];
  }
  return ret;
}



template <size_t N>
decltype(auto) hadd(StaticVec<N> const& val)
{
  union u_t
  {
    UnitVec const* u;
    StaticVec<N/2> const* h;
  };
  u_t const lhs {&val.data[0]};
  u_t const rhs {&val.data[StaticVec<N>::nb_unitvecs / 2]};
  return hadd(*lhs.h + *rhs.h);
}

template <>
decltype(auto) hadd(StaticVec<UnitVec::size> const& val)
{
  return val.data[0].hadd();
}

template <size_t N>
float dot_product__00(const float (&l)[N], const float (&r)[N])
{
  float ret{0};
  for (size_t i = 0; i < N; ++i)
  {
    ret += l[i] * r[i];
  }
  return ret;
}

template <size_t N>
float dot_product__01(StaticVec<N> const& l, StaticVec<N> const& r)
{
  float ret{0};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    ret += dot_product__01(l.data[i], r.data[i]);
  }
  return ret;
}

template <size_t N>
float dot_product__02(StaticVec<N> const& l, StaticVec<N> const& r)
{
  UnitVec acc{{0}};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    acc += l.data[i] * r.data[i];
  }
  return acc.hadd();
}

template <size_t N>
float dot_product__03(StaticVec<N>& l, StaticVec<N> const& r)
{
  float res{0};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    res += (l.data[i] * r.data[i]).hadd();
  }
  return res;
}


int main()
{
  constexpr size_t N = 1 << 13;
  StaticVec<N> u, v;
  float w[N], x[N];
  for (size_t i = 0; i < StaticVec<N>::size; ++i)
  {
    u[i] = v[StaticVec<N>::size - i - 1] = w[i] = x[N - i - 1] = i;
  }

  const auto start_res00 = clock_::now();
  const auto res00 = dot_product__00(w, x);
  const auto elapsed_res00 = to_ns(clock_::now() - start_res00);

  std::cout << "StaticVec 00 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res00
            << " elapsed " << elapsed_res00.count() << " ns"
            << std::endl;

  const auto start_res01 = clock_::now();
  const auto res01 = dot_product__01(u, v);
  const auto elapsed_res01 = to_ns(clock_::now() - start_res01);

  std::cout << "StaticVec 01 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res01
            << " elapsed " << elapsed_res01.count() << " ns"
            << std::endl;

  const auto start_res02 = clock_::now();
  const auto res02 = dot_product__02(u, v);
  const auto elapsed_res02 = to_ns(clock_::now() - start_res02);

  std::cout << "StaticVec 02 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res02
            << " elapsed " << elapsed_res02.count() << " ns"
            << std::endl;

  const auto start_res03 = clock_::now();
  const auto res03 = dot_product__03(u, v);
  const auto elapsed_res03 = to_ns(clock_::now() - start_res03);

  std::cout << "StaticVec 03 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res03
            << " elapsed " << elapsed_res03.count() << " ns"
            << std::endl;

  return 0;
}
