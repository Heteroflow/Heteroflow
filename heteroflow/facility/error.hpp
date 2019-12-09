#pragma once

#include <exception>
#include <system_error>
#include <cuda.h>

#include "macro.hpp"
#include "utility.hpp"

#define HF_THROW(...)                                 \
std::ostringstream oss;                               \
oss << "[" << __FILE__ << ":" << __LINE__ << "] ";    \
hf::stringify(oss, __VA_ARGS__);                      \
throw std::runtime_error(oss.str());                  \
}                                                     

#define HF_THROW_IF(...)                              \
if(HF_GET_FIRST(__VA_ARGS__)) {                       \
  std::ostringstream oss;                             \
  oss << "[" << __FILE__ << ":" << __LINE__ << "] ";  \
  hf::stringify(oss, HF_REMOVE_FIRST(__VA_ARGS__));   \
  throw std::runtime_error(oss.str());                \
}                                                     

#define HF_CHECK_CUDA(...)                            \
if(HF_GET_FIRST(__VA_ARGS__) != cudaSuccess) {        \
  std::ostringstream oss;                             \
  auto ev = HF_GET_FIRST(__VA_ARGS__);                \
  auto unknown_str  = "unknown error";                \
  auto unknown_name = "cudaErrorUnknown";             \
  auto error_str  = ::cudaGetErrorString(ev);         \
  auto error_name = ::cudaGetErrorName(ev);           \
  oss << "[" << __FILE__ << ":" << __LINE__ << "] "   \
      << (error_str  ? error_str  : unknown_str)      \
      << " ("                                         \
      << (error_name ? error_name : unknown_name)     \
      << ") - ";                                      \
  hf::stringify(oss, HF_REMOVE_FIRST(__VA_ARGS__));   \
  throw std::runtime_error(oss.str());                \
}




