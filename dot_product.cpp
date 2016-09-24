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

#include <Eigen/Core>


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
float dot_product__10(StaticVec<N> const& l, StaticVec<N> const& r)
{
  float ret{0};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    ret += dot_product__10(l.data[i], r.data[i]);
  }
  return ret;
}


template <size_t N>
float dot_product__20(StaticVec<N> const& l, StaticVec<N> const& r)
{
  UnitVec acc{{0}};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    acc += l.data[i] * r.data[i];
  }
  const auto ret = acc.hadd();
  return ret;
}


template <size_t N>
float dot_product__21(StaticVec<N> const& l, StaticVec<N> const& r)
{
  UnitVec acc0{{0}}, acc1{{0}};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs / 2; ++i)
  {
    acc0 += l.data[i * 2] * r.data[i * 2];
    acc1 += l.data[i * 2 + 1] * r.data[i * 2 + 1];
  }
  const auto ret = UnitVec{acc0 + acc1}.hadd();
  return ret;
}


template <size_t N>
float dot_product__22(StaticVec<N> const& l, StaticVec<N> const& r)

{
  UnitVec acc0{{0}}, acc1{{0}}, acc2{{0}}, acc3{{0}};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs / 4; ++i)
  {
    acc0 += l.data[i * 4] * r.data[i];
    acc1 += l.data[i * 4 + 1] * r.data[i * 4 + 1];
    acc2 += l.data[i * 4 + 2] * r.data[i * 4 + 2];
    acc3 += l.data[i * 4 + 3] * r.data[i * 4 + 3];

  }
  const auto ret = UnitVec{acc0 + acc1 + acc2 + acc3}.hadd();
  return ret;
}


template <size_t N>
float dot_product__30(StaticVec<N>& l, StaticVec<N> const& r)
{
  float res{0};
  for (size_t i = 0; i < StaticVec<N>::nb_unitvecs; ++i)
  {
    res += (l.data[i] * r.data[i]).hadd();
  }
  return res;
}

template <int N>
float dot_product__40(Eigen::Array<float, N, 1> const& l,
                      Eigen::Array<float, N, 1> const& r)
{
  const auto ret = (l + r).sum();
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

  const auto start_res00 = clock_::now();
  const auto res00 = dot_product__00(w, x);
  const auto elapsed_res00 = to_ns(clock_::now() - start_res00);

  std::cout << "StaticVec 00 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res00
            << " elapsed " << elapsed_res00.count() << " ns"
            << std::endl;

  const auto start_res10 = clock_::now();
  const auto res10 = dot_product__10(u, v);
  const auto elapsed_res10 = to_ns(clock_::now() - start_res10);

  std::cout << "StaticVec 10 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res10
            << " elapsed " <<elapsed_res10.count() << " ns"
            << std::endl;

  const auto start_res20 = clock_::now();
  const auto res20 = dot_product__20(u, v);
  const auto elapsed_res20 = to_ns(clock_::now() - start_res20);

  std::cout << "StaticVec 20 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res20
            << " elapsed " << elapsed_res20.count() << " ns"
            << std::endl;

  const auto start_res21 = clock_::now();
  const auto res21 = dot_product__21(u, v);
  const auto elapsed_res21 = to_ns(clock_::now() - start_res21);

  std::cout << "StaticVec 21 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res21
            << " elapsed " << elapsed_res21.count() << " ns"
            << std::endl;

  const auto start_res22 = clock_::now();
  const auto res22 = dot_product__22(u, v);
  const auto elapsed_res22 = to_ns(clock_::now() - start_res22);

  std::cout << "StaticVec 22 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res22
            << " elapsed " << elapsed_res22.count() << " ns"
            << std::endl;

  const auto start_res30 = clock_::now();
  const auto res30 = dot_product__30(u, v);
  const auto elapsed_res30 = to_ns(clock_::now() - start_res30);

  std::cout << "StaticVec 30 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res30
            << " elapsed " << elapsed_res30.count() << " ns"
            << std::endl;

  const auto start_res40 = clock_::now();
  const auto res40 = dot_product__40(y, z);
  const auto elapsed_res40 = to_ns(clock_::now() - start_res40);

  std::cout << "StaticVec 40 size " << StaticVec<N>::size
            << " nb_unitvecs " << StaticVec<N>::nb_unitvecs
            << " " << res40
            << " elapsed " << elapsed_res40.count() << " ns"
            << std::endl;

  return 0;
}
