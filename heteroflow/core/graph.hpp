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
  friend class TransferTask;
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
    int              device {-1};
    void*            d_data {nullptr};
    size_t           d_size {0};
  };  
  
  // Push data
  struct Push {
    Push() = default;
    std::function<void(cudaStream_t)> work;
    Node*        source {nullptr};
  };

  // Transfer data
  struct Transfer {
    Transfer() = default;
    std::function<void(cudaStream_t)> work;
    Node*        source {nullptr};
    Node*        target {nullptr};
  };
  
  // Kernel data
  struct Kernel {
    Kernel() = default;
    std::function<void(cudaStream_t)> work;
    int          device     {-1};
    ::dim3       grid       {1, 1, 1};
    ::dim3       block      {1, 1, 1}; 
    size_t       shm        {0};
    std::vector<Node*> sources;
  };

  struct DeviceGroup {
    std::atomic<int> device_id {-1};
    std::atomic<int> num_tasks {0};
  };
  
  public:

    template <typename... ArgsT>
    Node(ArgsT&&...);
    
    bool is_host() const;
    bool is_push() const;
    bool is_transfer() const;
    bool is_pull() const;
    bool is_kernel() const;
		bool is_device() const;

    void dump(std::ostream&) const;

    std::string dump() const;

    size_t num_successors() const;
    size_t num_dependents() const;

  private:

    static constexpr int HOST_IDX   = 0;
    static constexpr int PULL_IDX   = 1;
    static constexpr int PUSH_IDX   = 2;
    static constexpr int KERNEL_IDX = 3;
    static constexpr int TRANSFER_IDX  = 4;

    std::string _name;

    nstd::variant<Host, Pull, Push, Kernel, Transfer> _handle;

    std::vector<Node*> _successors;
    std::vector<Node*> _dependents;
    
    std::atomic<int> _num_dependents {0};

    Node* _parent {this};
    // TODO: use size instead of height, initial size value should be 1
    int   _tree_size {1};

		// Kernels in a group will be deployed on the same device
    DeviceGroup* _group {nullptr};
    
    Topology* _topology {nullptr};

    Node* _root();
    //Node* _parent() const;
    
    void _device(int);
    //void _height(int);
    //void _parent(Node*);
    void _union(Node*);
    void _precede(Node*);

    //int _height() const;
    int _device() const;

    Host& _host_handle();
    Pull& _pull_handle();
    Push& _push_handle();
    Transfer& _transfer_handle();
    Kernel& _kernel_handle();

    const Host& _host_handle() const;
    const Pull& _pull_handle() const;
    const Push& _push_handle() const;
    const Transfer& _transfer_handle() const;
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
  return nstd::get<Host>(_handle);
}

// Function: _host_handle    
inline const Node::Host& Node::_host_handle() const {
  return nstd::get<Host>(_handle);
}

// Function: _push_handle    
inline Node::Push& Node::_push_handle() {
  return nstd::get<Push>(_handle);
}

// Function: _push_handle    
inline const Node::Push& Node::_push_handle() const {
  return nstd::get<Push>(_handle);
}

// Function: _transfer_handle    
inline Node::Transfer& Node::_transfer_handle() {
  return nstd::get<Transfer>(_handle);
}

// Function: _transfer_handle    
inline const Node::Transfer& Node::_transfer_handle() const {
  return nstd::get<Transfer>(_handle);
}

// Function: _pull_handle    
inline Node::Pull& Node::_pull_handle() {
  return nstd::get<Pull>(_handle);
}

// Function: _pull_handle    
inline const Node::Pull& Node::_pull_handle() const {
  return nstd::get<Pull>(_handle);
}

// Function: _kernel_handle    
inline Node::Kernel& Node::_kernel_handle() {
  return nstd::get<Kernel>(_handle);
}

// Function: _kernel_handle    
inline const Node::Kernel& Node::_kernel_handle() const {
  return nstd::get<Kernel>(_handle);
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

// Function: is_transfer
inline bool Node::is_transfer() const {
  return _handle.index() == TRANSFER_IDX;
}

// Function: is_kernel
inline bool Node::is_kernel() const {
  return _handle.index() == KERNEL_IDX;
}

// Function: is_device
inline bool Node::is_device() const {
  return (is_push() || is_pull() || is_kernel() || is_transfer());
}

/*// Function: _height
inline int Node::_height() const {

  struct visitor {
    int operator () (const Host&   h) const { return -1; }
    int operator () (const Push&   h) const { return -1; }
    int operator () (const Pull&   h) const { return h.height; }
    int operator () (const Kernel& h) const { return h.height; }
  };

  return nstd::visit(visitor{}, _handle);
}

// Function: _parent
inline Node* Node::_parent() const {
  
  struct visitor {
    Node* operator () (const Host&   h) const { return nullptr; }
    Node* operator () (const Push&   h) const { return nullptr; }
    Node* operator () (const Pull&   h) const { return h.parent; }
    Node* operator () (const Kernel& h) const { return h.parent; }
  };

  return nstd::visit(visitor{}, _handle);
}

// Function: _height
inline void Node::_height(int h) {
  
  struct visitor {
    int v;
    void operator () (Host&   h) { }
    void operator () (Push&   h) { }
    void operator () (Pull&   h) { h.height = v; }
    void operator () (Kernel& h) { h.height = v; }
  };

  nstd::visit(visitor{h}, _handle);
}

// Function: _parent
inline void Node::_parent(Node* ptr) {
  
  struct visitor {
    Node* v;
    void operator () (Host&   h) { }
    void operator () (Push&   h) { }
    void operator () (Pull&   h) { h.parent = v; }
    void operator () (Kernel& h) { h.parent = v; }
  };

  nstd::visit(visitor{ptr}, _handle);
} */

// Function: _device
inline void Node::_device(int d) {

  struct visitor {
    int v;
    void operator () (Host&   h) { }
    void operator () (Push&   h) { }
    void operator () (Transfer&  h) { }
    void operator () (Pull&   h) { h.device = v; }
    void operator () (Kernel& h) { h.device = v; }
  };

  nstd::visit(visitor{d}, _handle);
}

// Function: _device
inline int Node::_device() const {

  struct visitor {
    int operator () (const Host&   h) const { return -1; }
    int operator () (const Push&   h) const { return -1; }
    int operator () (const Pull&   h) const { return h.device; }
    int operator () (const Kernel& h) const { return h.device; }
    int operator () (const Transfer&  h) const { return -1; }
  };

  return nstd::visit(visitor{}, _handle);
}

// Function: _root
inline Node* Node::_root() {
  auto ptr = this;
  while(ptr != _parent) {
    _parent = _parent->_parent; 
    ptr = _parent;
  }
  return ptr;
}

// TODO: use size instead of height
// Procedure: _union
inline void Node::_union(Node* y) {

  if(_parent == y->_parent) {
    return;
  }

  auto xroot = _root();
  auto yroot = y->_root();

  assert(xroot != yroot);

  auto xrank = xroot->_tree_size;
  auto yrank = yroot->_tree_size;

  if(xrank < yrank) {
    xroot->_parent = yroot;
    yroot->_tree_size += xrank;
  }
  else {
    yroot->_parent = xroot;
    xroot->_tree_size += yrank;
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

    // transfer
    case 4:
      os << " style=filled fillcolor=\"coral\"";
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






