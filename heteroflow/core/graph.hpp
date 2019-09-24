#pragma once

#include "../cuda/cuda.hpp"
#include "../facility/facility.hpp"

namespace hf {

// Forward declaration
class Topology;

// Class: Node
class Node {
    
  template <typename Drvied>
  friend class TaskBase;

  friend class HostTask;
  friend class PullTask;
  friend class PushTask;
  friend class KernelTask;
  
  friend class FlowBuilder;
  friend class Heteroflow;
  
  friend class Topology;
  friend class Executor;
  
  // Host data
  struct Host {
    std::function<void()> work;
  };
  
  // Pull data
  struct Pull {
    Pull() = default;
    std::function<void(cuda::Allocator&, cudaStream_t)> work;
    std::atomic<int> device {-1};
    void*            d_data {nullptr};
    size_t           d_size {0};
    int              height {0};
    Node*            parent {nullptr};
  };  
  
  // Push data
  struct Push {
    Push() = default;
    std::function<void(cudaStream_t)> work;
    Node*        source {nullptr};
  };
  
  // Kernel data
  struct Kernel {
    Kernel() = default;
    std::function<void(cudaStream_t)> work;
    std::atomic<int> device {-1};
    ::dim3       grid       {1, 1, 1};
    ::dim3       block      {1, 1, 1}; 
    size_t       shm        {0};
    std::vector<Node*> sources;
    int          height     {0};
    Node*        parent     {nullptr};
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

    size_t num_successors() const;
    size_t num_dependents() const;

  private:

    static constexpr int HOST_IDX   = 0;
    static constexpr int PULL_IDX   = 1;
    static constexpr int PUSH_IDX   = 2;
    static constexpr int KERNEL_IDX = 3;

    std::string _name;

    nonstd::variant<Host, Pull, Push, Kernel> _handle;

    std::vector<Node*> _successors;
    std::vector<Node*> _dependents;
    
    std::atomic<int> _num_dependents {0};
    
    Topology* _topology {nullptr};

    Node* _root();
    Node* _parent() const;
    
    void _height(int);
    void _parent(Node*);
    void _union(Node*);
    void _precede(Node*);

    int _height() const;

    std::atomic<int>& _device();

    Host& _host_handle();
    Pull& _pull_handle();
    Push& _push_handle();
    Kernel& _kernel_handle();

    const Host& _host_handle() const;
    const Pull& _pull_handle() const;
    const Push& _push_handle() const;
    const Kernel& _kernel_handle() const;
};

// ----------------------------------------------------------------------------
// Host field
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Pull field
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Push field
// ----------------------------------------------------------------------------

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

// Function: _host_handle    
inline const Node::Host& Node::_host_handle() const {
  return nonstd::get<Host>(_handle);
}

// Function: _push_handle    
inline Node::Push& Node::_push_handle() {
  return nonstd::get<Push>(_handle);
}

// Function: _push_handle    
inline const Node::Push& Node::_push_handle() const {
  return nonstd::get<Push>(_handle);
}

// Function: _pull_handle    
inline Node::Pull& Node::_pull_handle() {
  return nonstd::get<Pull>(_handle);
}

// Function: _pull_handle    
inline const Node::Pull& Node::_pull_handle() const {
  return nonstd::get<Pull>(_handle);
}

// Function: _kernel_handle    
inline Node::Kernel& Node::_kernel_handle() {
  return nonstd::get<Kernel>(_handle);
}

// Function: _kernel_handle    
inline const Node::Kernel& Node::_kernel_handle() const {
  return nonstd::get<Kernel>(_handle);
}

// Function: dump
inline std::string Node::dump() const {
  std::ostringstream os;  
  dump(os);
  return os.str();
}

// Function: num_successors
inline size_t Node::num_successors() const {
  return _successors.size();
}

// Function: num_dependents
inline size_t Node::num_dependents() const {
  return _dependents.size();
}

// Function: is_host
inline bool Node::is_host() const {
  return _handle.index() == HOST_IDX;
}

// Function: is_pull
inline bool Node::is_pull() const {
  return _handle.index() == PULL_IDX;
}

// Function: is_push
inline bool Node::is_push() const {
  return _handle.index() == PUSH_IDX;
}

// Function: is_kernel
inline bool Node::is_kernel() const {
  return _handle.index() == KERNEL_IDX;
}

// Function: _height
inline int Node::_height() const {
  int idx = _handle.index();
  if(idx == KERNEL_IDX) {
    return _kernel_handle().height;
  }
  else {
    assert(idx == PULL_IDX);
    return _pull_handle().height;
  }
}

// Function: _parent
inline Node* Node::_parent() const {
  int idx = _handle.index();
  if(idx == KERNEL_IDX) {
    return _kernel_handle().parent;
  }
  else {
    assert(idx == PULL_IDX);
    return _pull_handle().parent;
  }
}

// Function: _height
inline void Node::_height(int h) {
  int idx = _handle.index();
  if(idx == KERNEL_IDX) {
    _kernel_handle().height = h;
  }
  else {
    assert(idx == PULL_IDX);
    _pull_handle().height = h;
  }
}

// Function: _parent
inline void Node::_parent(Node* ptr) {
  int idx = _handle.index();
  if(idx == KERNEL_IDX) {
    _kernel_handle().parent = ptr;
  }
  else {
    assert(idx == PULL_IDX);
    _pull_handle().parent = ptr;
  }
}

// Function: _device
inline std::atomic<int>& Node::_device() {
  int idx = _handle.index();
  if(idx == KERNEL_IDX) {
    return _kernel_handle().device;
  }
  else {
    assert(idx == PULL_IDX);
    return _pull_handle().device;
  }
}

// Function: _root
inline Node* Node::_root() {
  Node* p = _parent();
  if(p == nullptr) {
    return this;
  }
  p = p->_root();
  _parent(p);
  return p;
}

// Procedure: _union
inline void Node::_union(Node* y) {

  auto xroot = _root();
  auto yroot = y->_root();
  auto xrank = xroot->_height();
  auto yrank = yroot->_height();

  if(xrank < yrank) {
    xroot->_parent(yroot);
  }
  else if(xrank > yrank) {
    yroot->_parent(xroot);
  }
  else {
    yroot->_parent(xroot);
    xroot->_height(xrank + 1);
  }
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



//// Function: _root
//inline Node* Node::_root() {
//
//  assert(is_kernel() || is_pull());
//
//  Node* parent {nullptr};
//
//  if(_parent == this) {
//    return this;
//  }
//
//  _parent = _parent->_root();
//
//  return _parent;
//}
//
//// Procedure: _union
//inline void Node::_union(Node* y) {
//
//  auto xroot = _root();
//  auto yroot = y->_root();
//
//  if(xroot->_rank < yroot->_rank) {
//    xroot->_parent = yroot;
//  }
//  else if(xroot->_rank > yroot->_rank) {
//    yroot->_parent = xroot;
//  }
//  else {
//    yroot->_parent = xroot;
//    ++xroot->_rank;
//  }
//}

/*// Function: root
inline Node* Node::_root(Node* node) {
  if(node->_parent != node) {
    node->_parent = _root(node->_parent);
  }
  return node->_parent;
}

// Procedure: union_node
inline void Node::_union_node(Node* x, Node* y) {

  auto xroot = _root(x);
  auto yroot = _root(y);

  if(xroot->_rank < yroot->_rank) {
    xroot->_parent = yroot;
  }
  else if(xroot->_rank > yroot->_rank) {
    yroot->_parent = xroot;
  }
  else {
    yroot->_parent = xroot;
    ++xroot->_rank;
  }
}*/

// ----------------------------------------------------------------------------


}  // end of namespace hf -----------------------------------------------------






