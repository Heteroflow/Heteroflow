#pragma once

#include "traits.hpp"
#include "error.hpp"
#include "notifier.hpp"
#include "spmc_queue.hpp"
#include "variant.hpp"
#include "optional.hpp"

// 3-rd party downgrade of C++17 libraries
namespace nonstd {
  
using namespace mpark;

}

// alias namespace
namespace nstd = nonstd;

