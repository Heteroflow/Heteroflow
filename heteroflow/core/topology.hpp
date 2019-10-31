#pragma once

#include "graph.hpp"

namespace hf {

// ----------------------------------------------------------------------------
// Forward declaration
// ----------------------------------------------------------------------------

class Heteroflow;

// ----------------------------------------------------------------------------
  
// class: Topology
class Topology {
  
  friend class Heteroflow;
  friend class Executor;
  
  public:

    template <typename P, typename C>
    Topology(Heteroflow&, P&&, C&&);
    
  private:

    Heteroflow& _heteroflow;

    std::promise<void> _promise;
    std::vector<Node*> _sources;

    std::atomic<int> _num_sinks {0};

    int _cached_num_sinks {0};
    
    std::function<bool()> _pred;
    std::function<void()> _call;
};

// Constructor
template <typename P, typename C>
inline Topology::Topology(Heteroflow& tf, P&& p, C&& c): 
  _heteroflow(tf),
  _pred {std::forward<P>(p)},
  _call {std::forward<C>(c)} {
}

}  // end of namespace tf. ----------------------------------------------------
