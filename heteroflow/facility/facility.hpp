#pragma once

// hf-specific enablement
#define span_FEATURE_MAKE_SPAN 1

#include "traits.hpp"
#include "error.hpp"
#include "notifier.hpp"
#include "spmc_queue.hpp"
#include "variant.hpp"
#include "optional.hpp"
#include "byte.hpp"
//#include "span.hpp"

// 3-rd party downgrade of C++17 libraries
namespace nonstd {
  
using namespace mpark;

}

namespace hf {

//using nonstd::make_span;

}

// alias namespace
namespace nstd = nonstd;

