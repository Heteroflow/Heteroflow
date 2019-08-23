#pragma once

#include "../facility/facility.hpp"

namespace hf {

// Class: Node
class Node {

  friend class TaskBase;
  friend class HostTask;
  friend class PullTask;
  friend class PushTask;
  friend class KernelTask;
  friend class FlowBuilder;
  
  // Host data
  struct Host {
  };
  
  // Pull data
  struct Pull {

    Pull() = default;

    template <typename T>
    Pull(const T*, size_t);

    ~Pull();

    int          device {0};
    cudaStream_t stream {0};
    const void*  h_data {nullptr};
    void*        d_data {nullptr};
    size_t       h_size {0};
    size_t       d_size {0};
  };  
  
  // Push data
  struct Push {

    Push() = default;

    template <typename T>
    Push(T*, Node*, size_t);
    
    int          device {0};
    cudaStream_t stream {0};
    void*        h_data {nullptr};
    Node*        source {nullptr};
    size_t       h_size {0};
  };
  
  // Kernel data
  struct Kernel {

    Kernel() = default;

    int          device {0};
    cudaStream_t stream {0};
    ::dim3       grid   {1, 1, 1};
    ::dim3       block  {1, 1, 1}; 
    size_t       shm    {0};
  };

  public:

    template <typename... ArgsT>
    Node(ArgsT&&...);
    
    bool is_host() const;
    bool is_push() const;
    bool is_pull() const;
    bool is_kernel() const;

  private:
    
    std::string _name;

    nonstd::variant<Host, Pull, Push, Kernel> _handle;

    std::function<void()> _work;

    std::vector<Node*> _successors;
    std::vector<Node*> _dependents;
    
    std::atomic<int> _num_dependents {0};

    void _precede(Node*);

    Host& _host_handle();
    Pull& _pull_handle();
    Push& _push_handle();
    Kernel& _kernel_handle();
};

// ----------------------------------------------------------------------------
// Host field
// ----------------------------------------------------------------------------

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
  h_size {N * sizeof(T)} {
}
    
// ----------------------------------------------------------------------------
// Kernel field
// ----------------------------------------------------------------------------

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

// Function: _host_handle    
inline Node::Host& Node::_host_handle() {
  return nonstd::get<Host>(_handle);
}

// Function: _push_handle    
inline Node::Push& Node::_push_handle() {
  return nonstd::get<Push>(_handle);
}

// Function: _pull_handle    
inline Node::Pull& Node::_pull_handle() {
  return nonstd::get<Pull>(_handle);
}

// Function: _kernel_handle    
inline Node::Kernel& Node::_kernel_handle() {
  return nonstd::get<Kernel>(_handle);
}

// ----------------------------------------------------------------------------


}  // end of namespace hf -----------------------------------------------------






