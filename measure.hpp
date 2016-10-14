#include <iostream>
#include <chrono>


using clock_ = std::chrono::high_resolution_clock;


template <class duration_t>
auto to_ns(duration_t&& val)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(val).count();
}

template <class func_t, class... Args>
auto measure(func_t func, Args&&... args)
{
  const auto start = clock_::now();
  auto res = func(std::forward<Args>(args)...);
  const auto stop = clock_::now();
  return std::make_pair(res, to_ns(stop - start));
}

template <class measure_t>
void print_measure(const char* method_name, measure_t&& measure)
{
  std::cout << "StaticVec " << method_name
            << " " << measure.first
            << " elapsed " << measure.second << " ns"
            << std::endl;
}


