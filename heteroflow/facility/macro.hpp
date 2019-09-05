#pragma once

#define HF_REMOVE_FIRST_HELPER(N, ...) __VA_ARGS__
#define HF_REMOVE_FIRST(...) HF_REMOVE_FIRST_HELPER(__VA_ARGS__)

#define HF_GET_FIRST_HELPER(N, ...) N
#define HF_GET_FIRST(...) HF_GET_FIRST_HELPER(__VA_ARGS__)

#define HF_FWD(...) std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)

#define HF_FOR_IF(item, container, expression) \
  for(auto& item : container) \
    if(expression)
