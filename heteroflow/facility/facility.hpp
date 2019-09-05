#pragma once

#include "traits.hpp"
#include "error.hpp"
#include "variant.hpp"
#include "optional.hpp"
#include "notifier.hpp"
#include "spmc_queue.hpp"

// 3-rd party downgrade of C++17 libraries
namespace nonstd {
  
using namespace mpark;

}
