project(dot_product_methods CXX)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

option(LLVM_BITCODE OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fno-rtti -fno-exceptions -mavx")
set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} -flto")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -ftree-vectorize -march=native -mtune=native")
else ()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fast -xHOST -ipo -qopt-report -qopt-report-phase=vec")
endif ()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(NOT LLVM_BITCODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} -lc++abi")
  endif()
  set(CMAKE_AR "llvm-ar")
  set(CMAKE_LD "llvm-ld")
  set(CMAKE_NM "llvm-nm" )
  set(CMAKE_OBJDUMP "llvm-objdump")
  set(CMAKE_RANLIB "llvm-ranlib")
endif()

if(LLVM_BITCODE)
  find_package(LLVM REQUIRED)
  include(${CMAKE_SOURCE_DIR}/cmake/LLVMTools.cmake)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-use-cxa-atexit")
  add_bitcode(dot_product dot_product.cpp)
  add_custom_target(bench
    COMMAND ${CMAKE_HOME_DIRECTORY}/scripts/bench.sh 1000 lli -O3 ${PROJECT_BINARY_DIR}/dot_product.bc
    DEPENDS dot_product)

else()
  add_executable(dot_product dot_product.cpp)
  add_custom_target(bench
    COMMAND ${CMAKE_HOME_DIRECTORY}/scripts/bench.sh 1000 ${PROJECT_BINARY_DIR}/dot_product
    DEPENDS dot_product)
endif()
