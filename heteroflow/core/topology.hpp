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

    void _bind(Graph& g);
    void _recover_num_sinks();
};

// Constructor
template <typename P, typename C>
inline Topology::Topology(Heteroflow& tf, P&& p, C&& c): 
  _heteroflow(tf),
  _pred {std::forward<P>(p)},
  _call {std::forward<C>(c)} {
}

// Procedure: _bind
// Re-builds the source links and the sink number for this topology.
inline void Topology::_bind(Graph& g) {
  
  _num_sinks = 0;
  _sources.clear();
  
  // scan each node in the graph and build up the links
  for(auto& node : g) {

    node->_topology = this;

    if(node->num_dependents() == 0) {
      _sources.push_back(node.get());
    }

    if(node->num_successors() == 0) {
      _num_sinks++;
    }
    
    if(node->is_kernel()) {
      auto& h = node->_kernel_handle();
      assert(h.device == -1);
      for(auto s : h.sources) {
        node->_union(s);
      }
    }
  }

  _cached_num_sinks = _num_sinks;
  
  // path compression
  for(auto& node : g) {
    if(node->is_kernel() || node->is_pull()) {
      std::cout << node->_name << ' ' << node->_root()->_name << std::endl;
    }
  }
}

// Procedure: _recover_num_sinks
inline void Topology::_recover_num_sinks() {
  _num_sinks = _cached_num_sinks;
}

}  // end of namespace tf. ----------------------------------------------------
