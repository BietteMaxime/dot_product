project(dot_product_methods CXX)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Werror -pedantic -fno-rtti -fno-exceptions -msse4.1 -mno-avx")
set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} -flto")

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -ftree-vectorize")

add_executable(dot_product dot_product.cpp)
