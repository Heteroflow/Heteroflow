#pragma once

#include <tuple>

namespace hf {

template<typename F>
struct function_traits;
 
// function pointer
template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> : public function_traits<R(Args...)> {
};

// function reference
template<typename R, typename... Args>
struct function_traits<R(&)(Args...)> : public function_traits<R(Args...)> {
};
 
 // function_traits
template<typename R, typename... Args>
struct function_traits<R(Args...)> {

  using return_type = R;
 
  static constexpr std::size_t arity = sizeof...(Args);
 
  template <std::size_t N>
  struct argument {
    static_assert(N < arity, "error: invalid parameter index.");
    using type = typename std::tuple_element<N,std::tuple<Args...>>::type;
  };
};

}  // end of namespace hf -----------------------------------------------------
