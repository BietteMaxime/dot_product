#include <cassert>

#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#elif defined(__SSE3__)
#include <pmmintrin.h>
#endif

#include <Eigen/Core>

#include "measure.hpp"


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
} __attribute__((aligned(16), packed));

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

float dot_product__10(UnitVec const& a, UnitVec const& b)
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

  static constexpr auto idx_(size_t idx)
  {
    return std::div(static_cast<long>(idx),
                    static_cast<long>(UnitVec::size));
  }

  float const& operator[](size_t idx) const
  {
    assert(idx < N);
    const auto i = idx_(idx);
    return data[i.quot][i.rem];
  }

  float& operator[](size_t idx)
  {
    assert(idx < N);
    const auto i = idx_(idx);
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
__attribute__((noinline))
float dot_product__00(const float (&l)[N], const float (&r)[N])
{
  asm("#dot_product__00_start");
  float ret{0};
  for (size_t i = 0; i < N; ++i)
  {
    ret += l[i] * r[i];
  }
  asm("#dot_product__00_end");
  return ret;
}


template <size_t N>
__attribute__((noinline))
float dot_product__10(StaticVec<N> const& l, StaticVec<N> const& r)
{
  asm("#dot_product__10_start");
  float ret{0};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    ret += dot_product__10(l.data[i], r.data[i]);
  }
  asm("#dot_product__10_end");
  return ret;
}


template <size_t N>
__attribute__((noinline))
float dot_product__20(StaticVec<N> const& l, StaticVec<N> const& r)
{
  asm("#dot_product__20_start");
  UnitVec acc{{0}};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    acc += l.data[i] * r.data[i];
  }
  const auto ret = acc.hadd();
  asm("#dot_product__20_end");
  return ret;
}


template <size_t N>
__attribute__((noinline))
float dot_product__21(StaticVec<N> const& l, StaticVec<N> const& r)
{
  asm("#dot_product__21_start");
  UnitVec acc0{{0}}, acc1{{0}};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs / 2; ++i)
  {
    acc0 += l.data[i * 2] * r.data[i * 2];
    acc1 += l.data[i * 2 + 1] * r.data[i * 2 + 1];
  }
  const auto ret = UnitVec{acc0 + acc1}.hadd();
  asm("#dot_product__21_end");
  return ret;
}


template <size_t N>
__attribute__((noinline))
float dot_product__22(StaticVec<N> const& l, StaticVec<N> const& r)

{
  asm("#dot_product__22_start");
  UnitVec acc0{{0}}, acc1{{0}}, acc2{{0}}, acc3{{0}};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs / 4; ++i)
  {
    acc0 += l.data[i * 4] * r.data[i];
    acc1 += l.data[i * 4 + 1] * r.data[i * 4 + 1];
    acc2 += l.data[i * 4 + 2] * r.data[i * 4 + 2];
    acc3 += l.data[i * 4 + 3] * r.data[i * 4 + 3];

  }
  const auto ret = UnitVec{acc0 + acc1 + acc2 + acc3}.hadd();
  asm("#dot_product__22_end");
  return ret;
}


template <size_t N>
__attribute__((noinline))
float dot_product__30(StaticVec<N> const& l, StaticVec<N> const& r)
{
  asm("#dot_product__30_start");
  float res{0};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    res += (l.data[i] * r.data[i]).hadd();
  }
  asm("#dot_product__30_end");
  return res;
}

template <int N>
__attribute__((noinline))
float dot_product__40(Eigen::Array<float, N, 1> const& l,
                      Eigen::Array<float, N, 1> const& r)
{
  asm("#dot_product__40_start");
  const auto ret = (l + r).sum();
  asm("#dot_product__40_end");
  return ret;
}


int main()
{
  constexpr size_t N = 1 << 13;
  StaticVec<N> u, v;
  float w[N], x[N];
  Eigen::Array<float, N, 1> y, z;
  for (size_t i = 0; i < StaticVec<N>::size; ++i)
  {
    u[i] = v[StaticVec<N>::size - i - 1]
      = w[i] = x[N - i - 1]
      = y[i] = z[N - i - 1]
      = i;
  }


  const auto res00 = measure([] (const float (&a)[N], const float(&b)[N]) {
        return dot_product__00(a, b);
      }, w, x);
  print_measure("00", res00);

  const auto res10 = measure([] (const StaticVec<N>& a,
                                 const StaticVec<N>& b) {
      return dot_product__10(a, b);
    }, u, v);
  print_measure("10", res10);

  const auto res20 = measure([] (const StaticVec<N>& a,
                                 const StaticVec<N>& b) {
      return dot_product__20(a, b);
    }, u, v);
  print_measure("20", res20);

  const auto res21 = measure([] (const StaticVec<N>& a,
                                 const StaticVec<N>& b) {
      return dot_product__21(a, b);
    }, u, v);
  print_measure("21", res21);

  const auto res22 = measure([] (const StaticVec<N>& a,
                                 const StaticVec<N>& b) {
      return dot_product__22(a, b);
    }, u, v);
  print_measure("22", res22);

  const auto res30 = measure([] (const StaticVec<N>& a,
                                 const StaticVec<N>& b) {
      return dot_product__30(a, b);
    }, u, v);
  print_measure("30", res30);

  const auto res40 = measure([] (const Eigen::Array<float, N, 1>& a,
                                 const Eigen::Array<float, N, 1>& b) {
      return dot_product__40(a, b);
    }, y, z);
  print_measure("40", res40);

  return 0;
}
