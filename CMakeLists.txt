cmake_minimum_required(VERSION 3.16)
project(chromatic_rl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# GMP (C and C++ bindings) - exact bignum coefficients.
find_library(GMP_LIB gmp REQUIRED)
find_library(GMPXX_LIB gmpxx REQUIRED)

add_executable(validate tests/validate.cpp)
target_include_directories(validate PRIVATE include)
target_link_libraries(validate PRIVATE ${GMPXX_LIB} ${GMP_LIB})

enable_testing()
add_test(NAME validate COMMAND validate)

# Optional: nauty-backed isomorphism cache. Built only if nauty is found.
# Debian/Ubuntu: header is <nauty/nauty.h>; package names vary (e.g. libnauty2-dev).
# Source build: https://pallini.di.uniroma1.it/  (then headers may be plain "nauty.h").
#
# Prefer a vendored nauty checked out under ${CMAKE_SOURCE_DIR}/nauty (built in place,
# producing nauty.a). canon.hpp includes <nauty/nauty.h>, so the repo root is the include
# dir. Fall back to a system-installed nauty if the vendored copy is absent.
#
# nauty itself is NOT bundled with this repo (see build_nauty.sh + README): fetch it
# from https://pallini.di.uniroma1.it/ (e.g. nauty2_9_3.tar.gz), drop the tarball at the
# repo root, then run ./build_nauty.sh, which builds it portable + -fPIC automatically.
file(GLOB _nauty_tarball "${CMAKE_SOURCE_DIR}/nauty*.tar.gz")
if(EXISTS "${CMAKE_SOURCE_DIR}/build_nauty.sh"
   AND (EXISTS "${CMAKE_SOURCE_DIR}/nauty/configure" OR _nauty_tarball))
  message(STATUS "Ensuring vendored nauty.a is built (portable, -fPIC) via build_nauty.sh")
  execute_process(
    COMMAND bash "${CMAKE_SOURCE_DIR}/build_nauty.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _nauty_build_rc)
  if(NOT _nauty_build_rc EQUAL 0)
    message(WARNING "build_nauty.sh failed (rc=${_nauty_build_rc}); falling back to system nauty if available")
  endif()
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/nauty/nauty.h" AND EXISTS "${CMAKE_SOURCE_DIR}/nauty/nauty.a")
  set(NAUTY_INCLUDE_DIR "${CMAKE_SOURCE_DIR}")
  set(NAUTY_LIB "${CMAKE_SOURCE_DIR}/nauty/nauty.a")
else()
  find_path(NAUTY_INCLUDE_DIR NAMES nauty/nauty.h nauty.h)
  find_library(NAUTY_LIB NAMES nauty)
endif()
if(NAUTY_INCLUDE_DIR AND NAUTY_LIB)
  message(STATUS "nauty found: ${NAUTY_LIB} (include: ${NAUTY_INCLUDE_DIR})")
  add_executable(bench_cache tests/bench_cache.cpp)
  target_include_directories(bench_cache PRIVATE include ${NAUTY_INCLUDE_DIR})
  target_link_libraries(bench_cache PRIVATE ${NAUTY_LIB} ${GMPXX_LIB} ${GMP_LIB})
  add_test(NAME bench_cache COMMAND bench_cache)
else()
  message(STATUS "nauty NOT found. Skipping bench_cache. Install nauty dev files to build it.")
endif()

# Optional: Python bindings. Built only when pybind11 is available (i.e. when invoked
# through scikit-build-core / pip). The plain `cmake -B build` C++ workflow skips this,
# just like the nauty block above.
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND AND NAUTY_INCLUDE_DIR AND NAUTY_LIB)
  message(STATUS "pybind11 found. building the chromatic_rl._core extension")
  pybind11_add_module(_core bindings/chromatic_py.cpp)
  target_include_directories(_core PRIVATE include ${NAUTY_INCLUDE_DIR})
  target_link_libraries(_core PRIVATE ${NAUTY_LIB} ${GMPXX_LIB} ${GMP_LIB})
  install(TARGETS _core DESTINATION chromatic_rl)
else()
  message(STATUS "pybind11 not found. skipping Python bindings")
endif()
