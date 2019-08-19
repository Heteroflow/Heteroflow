#pragma once

#include <iostream>
#include <string>
#include <atomic>
#include <vector>
#include <functional>

#include "../facility/variant.hpp"
#include "../facility/error.hpp"

namespace hf {

class Node {

  friend class TaskBase;
  friend class FlowBuilder;
  
  // Host data
  struct Host {

    template <typename C>
    Host(C&&);

    std::function<void()> work;
  };
  
  // Pull data
  struct Pull {

    template <typename T>
    Pull(const T*, size_t);

    ~Pull();

    int device {0};
    const void* h_data {nullptr};
    void*  d_data {nullptr};
    size_t h_size {0};
    size_t d_size {0};
  };
  
  // Push data
  struct Push {

    template <typename T>
    Push(T*, Node*, size_t);

    void* h_data {nullptr};
    Node* source {nullptr};
    size_t h_size {0};
  };
  
  // Kernel data
  struct Kernel {
  };

  public:

    Node() = default;

    template <typename... ArgsT>
    Node(ArgsT&&...);
    
    bool is_host() const;
    bool is_push() const;
    bool is_pull() const;
    bool is_kernel() const;

  private:
    
    std::string _name;

    mpark::variant<Host, Pull, Push, Kernel> _handle;

    std::vector<Node*> _successors;
    std::vector<Node*> _dependents;
    
    std::atomic<int> _num_dependents {0};

    void _precede(Node*);
};

// ----------------------------------------------------------------------------
// Host field
// ----------------------------------------------------------------------------

// Constructor
template <typename C>
Node::Host::Host(C&& callable) : 
  work {std::forward<C>(callable)} {
}

// ----------------------------------------------------------------------------
// Pull field
// ----------------------------------------------------------------------------

// Constructor
template <typename T>
Node::Pull::Pull(const T* data, size_t N) : 
  h_data {data},
  h_size {N * sizeof(T)} {
}

// Destructor
inline Node::Pull::~Pull() {
  if(d_data) {
    ::cudaSetDevice(device);
    ::cudaFree(d_data);
  }
}

// ----------------------------------------------------------------------------
// Push field
// ----------------------------------------------------------------------------

// Constructor
template <typename T>
Node::Push::Push(T* tgt, Node* src, size_t N) : 
  h_data {tgt},
  source {src},
  h_size {N} {
}

// ----------------------------------------------------------------------------

// Constructor
template <typename... ArgsT>
Node::Node(ArgsT&&... args) : 
  _handle {std::forward<ArgsT>(args)...} {
}

// Procedure: _precede
inline void Node::_precede(Node* rhs) {
  _successors.push_back(rhs);
  rhs->_dependents.push_back(this);
  rhs->_num_dependents.fetch_add(1, std::memory_order_relaxed);
}

// ----------------------------------------------------------------------------


}  // end of namespace hf -----------------------------------------------------






