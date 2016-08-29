project(dot_product_methods CXX)
cmake_minimum_required(VERSION 2.8)

option(LLVM_BITCODE OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Werror -pedantic -fno-rtti -fno-exceptions -mavx")
set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} -flto")

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -ftree-vectorize")

if(LLVM_BITCODE)
  find_package(LLVM REQUIRED)
  include(${CMAKE_SOURCE_DIR}/cmake/LLVMTools.cmake)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-use-cxa-atexit")
  add_bitcode(dot_product dot_product.cpp)
else()
  add_executable(dot_product dot_product.cpp)
endif()
