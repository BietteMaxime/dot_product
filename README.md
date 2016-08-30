# Dot product computation bench

## Build

As a cmake project, create a build directory and invoke cmake.
Has been tested with Gcc 6.1.1, Clang 3.8.1.

Examples:

- Vanilla compiler: `cmake .. -DCMAKE_BUILD_TYPE=Release`
- Clang: `CXX=clang++ cmake .. -DCMAKE_BUILD_TYPE=Release`
- Clang+LLVM: `CXX=clang++ cmake .. -DCMAKE_BUILD_TYPE=Release -DLLVM_BITCODE=1`

## Run

make bench