#pragma once

#include <type_traits>
#include <iterator>
#include <iostream>
#include <mutex>
#include <deque>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <list>
#include <forward_list>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <cmath>

namespace hf {

// Procedure: stringify
template <typename T>
void stringify(std::ostringstream& oss, T&& token) {
  oss << std::forward<T>(token);  
}

// Procedure: stringify
template <typename T, typename... Rest>
void stringify(std::ostringstream& oss, T&& token, Rest&&... rest) {
  oss << std::forward<T>(token);
  stringify(oss, std::forward<Rest>(rest)...);
}

// Procedure: va_count
template<typename ...Args>
constexpr std::size_t va_count(Args&&...) { 
  return sizeof...(Args); 
}

// PointerCaster
struct PointerCaster {

  void* d_data {nullptr};
  
  template <typename T>  
  operator T* ();
};

template <typename T>
PointerCaster::operator T*() {
  return (T*) d_data;
}

}  // end of namespace hf -----------------------------------------------------
