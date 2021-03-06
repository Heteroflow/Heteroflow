# CMake version
cmake_minimum_required (VERSION 3.8)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
project(Heteroflow VERSION 0.1.0 LANGUAGES CUDA CXX)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------
# testing
include(CTest)

# pthread
include(FindThreads)

# cuda
find_package(CUDA 9.0 REQUIRED)

# -----------------------------------------------------------------------------
# defult release build
# -----------------------------------------------------------------------------
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}'")
  set(
    CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" 
    CACHE
    STRING "Choose the type of build." 
    FORCE
  )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE 
    CMAKE_BUILD_TYPE 
    PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

# CXX target properties
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# project-specific variables
# -----------------------------------------------------------------------------
# build options
option(HF_BUILD_EXAMPLES "Enables build of examples" ON)
option(HF_BUILD_TESTS "Enables build of tests" ON)
option(HF_BUILD_BENCHMARKS "Enables build of benchmarks" OFF)

# installation path
set(HF_INC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(HF_LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(HF_UTEST_DIR ${PROJECT_SOURCE_DIR}/unittests)
set(HF_EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples)
set(HF_BENCHMARK_DIR ${PROJECT_SOURCE_DIR}/benchmarks)
set(HF_THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

# -----------------------------------------------------------------------------
# message
# -----------------------------------------------------------------------------
message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})
message(STATUS "CMAKE_HOST_SYSTEM: ${CMAKE_HOST_SYSTEM}")
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})
message(STATUS "CMAKE_CUDA_COMPILER: " ${CMAKE_CUDA_COMPILER})
message(STATUS "CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: " ${CMAKE_CUDA_FLAGS_DEBUG})
message(STATUS "CMAKE_CUDA_FLAGS_RELEASE: " ${CMAKE_CUDA_FLAGS_RELEASE})
message(STATUS "CUDA_VERSION: " ${CUDA_VERSION})
message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_LIBRARIES: " ${CUDA_LIBRARIES})
message(STATUS "CUDA_HAS_FP16: " ${CUDA_HAS_FP16})
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: " ${CUDA_TOOLKIT_ROOT_DIR})
message(STATUS "CUDA_SDK_ROOT_DIR: " ${CUDA_SDK_ROOT_DIR})
message(STATUS "BUILD_EXAMPLES: " ${HF_BUILD_EXAMPLES})
message(STATUS "BUILD_TESTS: " ${HF_BUILD_TESTS})
message(STATUS "BUILD_BENCHMARKS: " ${HF_BUILD_BENCHMARKS})
message(STATUS "INC_INSTALL_DIR: " ${HF_INC_INSTALL_DIR})
message(STATUS "LIB_INSTALL_DIR: " ${HF_LIB_INSTALL_DIR})
message(STATUS "UTEST_DIR: " ${HF_UTEST_DIR})
message(STATUS "EXAMPLE_DIR: " ${HF_EXAMPLE_DIR})
message(STATUS "BENCHMARK_DIR: " ${HF_BENCHMARK_DIR})

# -----------------------------------------------------------------------------
# add the binary tree to the search path for include files
# -----------------------------------------------------------------------------
include_directories(${PROJECT_SOURCE_DIR})

# -----------------------------------------------------------------------------
# Heteroflow library interface
# -----------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 14)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/> 
)
target_compile_options(${PROJECT_NAME} INTERFACE
  # CXX
  $<$<COMPILE_LANGUAGE:CXX>:$<BUILD_INTERFACE:-Wall>>
  # CUDA
  $<$<COMPILE_LANGUAGE:CUDA>:$<BUILD_INTERFACE:-Xcompiler=-Wall,-Wextra,-Wfatal-errors>>
)

# -----------------------------------------------------------------------------
# Example program 
# -----------------------------------------------------------------------------

if(${HF_BUILD_EXAMPLES})

message(STATUS "Building examples ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HF_EXAMPLE_DIR})

add_executable(saxpy ${HF_EXAMPLE_DIR}/saxpy.cu)
#set_property(TARGET saxpy PROPERTY CUDA_STANDARD 14)
target_link_libraries(saxpy ${CUDA_LIBRARIES} ${PROJECT_NAME})

add_executable(matrix-multiplication ${HF_EXAMPLE_DIR}/matrix-multiplication.cu)
#set_property(TARGET matrix-multiplication PROPERTY CUDA_STANDARD 14)
target_link_libraries(matrix-multiplication ${CUDA_LIBRARIES} ${PROJECT_NAME})

endif()

# -----------------------------------------------------------------------------
# Unittest
# -----------------------------------------------------------------------------

if(${HF_BUILD_TESTS})
 
enable_testing()
message(STATUS "Building unit tests ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HF_UTEST_DIR})

# unittest for basics
add_executable(basics unittests/basics.cu)
target_link_libraries(basics ${CUDA_LIBRARIES} ${PROJECT_NAME})
target_include_directories(basics PRIVATE ${HF_THIRD_PARTY_DIR}/doctest)
add_test(basics.static ${HF_UTEST_DIR}/basics -tc=static)
add_test(basics.host-tasks ${HF_UTEST_DIR}/basics -tc=host-tasks)
add_test(basics.span ${HF_UTEST_DIR}/basics -tc=span)
add_test(basics.memset ${HF_UTEST_DIR}/basics -tc=memset)
add_test(basics.d2d ${HF_UTEST_DIR}/basics -tc=d2d)
add_test(basics.h2d ${HF_UTEST_DIR}/basics -tc=h2d)
add_test(basics.d2h ${HF_UTEST_DIR}/basics -tc=d2h)
add_test(basics.h2d2h ${HF_UTEST_DIR}/basics -tc=h2d2h)
add_test(basics.h2d2d2h ${HF_UTEST_DIR}/basics -tc=h2d2d2h)
add_test(basics.dependent-copies ${HF_UTEST_DIR}/basics -tc=dependent-copies)
add_test(basics.chained-kernels ${HF_UTEST_DIR}/basics -tc=chained-kernels)
add_test(basics.dependent-kernels ${HF_UTEST_DIR}/basics -tc=dependent-kernels)
add_test(basics.statefulness ${HF_UTEST_DIR}/basics -tc=statefulness)
add_test(basics.run_n ${HF_UTEST_DIR}/basics -tc=run_n)

# unittest for matrix-ops
add_executable(matrix-ops unittests/matrix-ops.cu)
target_link_libraries(matrix-ops ${CUDA_LIBRARIES} ${PROJECT_NAME})
target_include_directories(matrix-ops PRIVATE ${HF_THIRD_PARTY_DIR}/doctest)
add_test(matrix.multiplication ${HF_UTEST_DIR}/matrix-ops -tc=multiplication)
add_test(matrix.transpose ${HF_UTEST_DIR}/matrix-ops -tc=transpose)
add_test(matrix.product ${HF_UTEST_DIR}/matrix-ops -tc=product)

endif()

# -----------------------------------------------------------------------------
# Benchmarking (enabled by BUILD_BENCHMARKS)
# -----------------------------------------------------------------------------

if(${HF_BUILD_BENCHMARKS})

## eigen package 
#if(NOT DEFINED EIGEN_ROOT)
#  set(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/3rd-party/eigen-3.3.7)
#endif()
#include_directories(${EIGEN_ROOT})
#
## find OpenMP package
#include(FindOpenMP)
#
#if(NOT OpenMP_CXX_FOUND)
#  message(FATAL_ERROR "OpenMP not found")
#endif()
#  
#message(STATUS "OpenMP_VERSION: ${OpenMP_VERSION}")
#message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
#message(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
#
## tbb package
#if(NOT DEFINED TBB_ROOT)
#  set(TBB_ROOT ${PROJECT_SOURCE_DIR}/3rd-party/tbb)
#endif()
#
#message(STATUS "TBB_ROOT: " ${TBB_ROOT})
#
#include(${TBB_ROOT}/cmake/TBBBuild.cmake)
#tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS tbb_cpf=1)
#find_package(TBB REQUIRED tbb_preview)
#
### benchmark 1: wavefront computing
#message(STATUS "benchmark 1: wavefront")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_DIR}/wavefront)
#add_executable(
#  wavefront 
#  ${BENCHMARK_DIR}/wavefront/main.cpp
#  ${BENCHMARK_DIR}/wavefront/omp.cpp
#  ${BENCHMARK_DIR}/wavefront/tbb.cpp
#  ${BENCHMARK_DIR}/wavefront/seq.cpp
#  ${BENCHMARK_DIR}/wavefront/taskflow.cpp
#)
#target_include_directories(wavefront PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
#target_link_libraries(
#  wavefront 
#  ${PROJECT_NAME} 
#  Threads::Threads 
#  ${TBB_IMPORTED_TARGETS} 
#  ${OpenMP_CXX_LIBRARIES} 
#  hf::default_settings
#)
#set_target_properties(wavefront PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

endif()

