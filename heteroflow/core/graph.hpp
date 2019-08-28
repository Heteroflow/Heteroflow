#pragma once

#include "../facility/facility.hpp"

namespace hf {

// Class: Node
class Node {
    
  template <typename Drvied>
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
    Pull(const void*, size_t);

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
    Push(void*, Node*, size_t);
    
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

    std::vector<Node*> sources;
  };

  public:

    template <typename... ArgsT>
    Node(ArgsT&&...);
    
    bool is_host() const;
    bool is_push() const;
    bool is_pull() const;
    bool is_kernel() const;

    void dump(std::ostream&) const;

    std::string dump() const;

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
inline Node::Pull::Pull(const void* data, size_t N) : 
  h_data {data},
  h_size {N} {
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
inline Node::Push::Push(void* tgt, Node* src, size_t N) : 
  h_data {tgt},
  source {src},
  h_size {N} {
}
    
// ----------------------------------------------------------------------------
// Kernel field
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Node field
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

// Function: dump
inline std::string Node::dump() const {
  std::ostringstream os;  
  dump(os);
  return os.str();
}

// Function: dump
inline void Node::dump(std::ostream& os) const {

  os << 'p' << this << "[label=\"";
  if(_name.empty()) {
    os << 'p' << this << "\"";
  }
  else {
    os << _name << "\"";
  }

  // color
  switch(_handle.index()) {
    // pull
    case 1:
      os << " style=filled fillcolor=\"cyan\"";
    break;
    
    // push
    case 2:
      os << " style=filled fillcolor=\"springgreen\"";
    break;
    
    // kernel
    case 3:
      os << " style=filled fillcolor=\"black\" fontcolor=\"white\" shape=\"diamond\"";
    break;

    default:
    break;
  };

  os << "];\n";
  
  for(const auto s : _successors) {
    os << 'p' << this << " -> " << 'p' << s << ";\n";
  }
}



}  // end of namespace hf -----------------------------------------------------






