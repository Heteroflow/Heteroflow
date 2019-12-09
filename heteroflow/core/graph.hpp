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
  friend class SpanTask;
  friend class CopyTask;
  friend class FillTask;
  friend class KernelTask;
  
  friend class FlowBuilder;
  friend class Heteroflow;
  
  friend class Topology;
  friend class Executor;
  
  // Host data
  struct Host {
    std::function<void()> work;
  };

  // Span data
  struct Span {
    Span() = default;
    std::function<void(cuda::Allocator&, cudaStream_t)> work;
    int device {-1};
    void* d_data {nullptr};
    size_t d_size {0};
  };  
  
  // Copy data
  struct Copy {
    Copy() = default;
    std::function<void(cudaStream_t)> work;
    Node* span {nullptr};
    cudaMemcpyKind direction {cudaMemcpyDefault};
  };

  // Fill data
  struct Fill {
    Fill() = default;
    std::function<void(cudaStream_t)> work;
    Node* span {nullptr};
  };
  
  // Kernel data
  struct Kernel {
    Kernel() = default;
    std::function<void(cudaStream_t)> work;
    int device {-1};
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
    bool is_copy() const;
    bool is_fill() const;
    bool is_span() const;
    bool is_kernel() const;
		bool is_device() const;

    void dump(std::ostream&) const;

    std::string dump() const;

    size_t num_successors() const;
    size_t num_dependents() const;

  private:

    static constexpr int HOST_IDX   = 0;
    static constexpr int SPAN_IDX   = 1;
    static constexpr int COPY_IDX   = 2;
    static constexpr int KERNEL_IDX = 3;
    static constexpr int FILL_IDX   = 4;

    std::string _name;

    nstd::variant<Host, Span, Copy, Kernel, Fill> _handle;

    std::vector<Node*> _successors;
    std::vector<Node*> _dependents;
    
    std::atomic<int> _num_dependents {0};

    Node* _parent {this};
    int   _tree_size {1};

		// Kernels in a group will be deployed on the same device
    DeviceGroup* _group {nullptr};
    
    Topology* _topology {nullptr};

    Node* _root();
    
    void _union(Node*);
    void _precede(Node*);

    Host& _host_handle();
    Span& _span_handle();
    Copy& _copy_handle();
    Fill& _fill_handle();
    Kernel& _kernel_handle();

    const Host& _host_handle() const;
    const Span& _span_handle() const;
    const Copy& _copy_handle() const;
    const Fill& _fill_handle() const;
    const Kernel& _kernel_handle() const;
};

// ----------------------------------------------------------------------------
// Host field
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Span field
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Copy field
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

// Function: _copy_handle    
inline Node::Copy& Node::_copy_handle() {
  return nstd::get<Copy>(_handle);
}

// Function: _copy_handle    
inline const Node::Copy& Node::_copy_handle() const {
  return nstd::get<Copy>(_handle);
}

// Function: _fill_handle    
inline Node::Fill& Node::_fill_handle() {
  return nstd::get<Fill>(_handle);
}

// Function: _fill_handle    
inline const Node::Fill& Node::_fill_handle() const {
  return nstd::get<Fill>(_handle);
}

// Function: _span_handle    
inline Node::Span& Node::_span_handle() {
  return nstd::get<Span>(_handle);
}

// Function: _span_handle    
inline const Node::Span& Node::_span_handle() const {
  return nstd::get<Span>(_handle);
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

// Function: is_span
inline bool Node::is_span() const {
  return _handle.index() == SPAN_IDX;
}

// Function: is_copy
inline bool Node::is_copy() const {
  return _handle.index() == COPY_IDX;
}

// Function: is_fill
inline bool Node::is_fill() const {
  return _handle.index() == FILL_IDX;
}

// Function: is_kernel
inline bool Node::is_kernel() const {
  return _handle.index() == KERNEL_IDX;
}

// Function: is_device
inline bool Node::is_device() const {
  return (is_copy() || is_span() || is_kernel() || is_fill());
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
    // span
    case 1:
      os << " style=filled fillcolor=\"cyan\"";
    break;
    
    // copy
    case 2: {
      auto& h = _copy_handle();
      switch(h.direction) {
        case cudaMemcpyHostToDevice:
          os << " style=filled fillcolor=\"orange\"";
        break;

        case cudaMemcpyDeviceToHost:
          os << " style=filled fillcolor=\"yellow\"";
        break;

        default:
          os << " style=filled fillcolor=\"springgreen\"";
        break;
      }
    }
    break;
    
    // kernel
    case 3:
      os << " style=\"filled\""
         << " fillcolor=\"black\""
         << " fontcolor=\"white\""
         << " shape=\"box3d\"";
    break;

    // fill
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

}  // end of namespace hf -----------------------------------------------------






/*// Procedure: _d2h
inline void CopyTask::_d2h(
  void* tgt, SpanTask src, size_t offset, size_t bytes, cudaStream_t stream
) {

  auto& s = src._node->_span_handle();

  auto ptr = static_cast<char*>(s.d_data) + offset;

  HF_CHECK_CUDA(
    cudaMemcpyAsync(
      tgt, ptr, bytes, cudaMemcpyDeviceToHost, stream
    ),
    "copy (D2H) task '", name(), "' failed\n",
    "target host/size: ", tgt, '/', bytes, '\n',
    "source span/size/offset: ", s.d_data, '/', s.d_size, '/', offset
  );
}

// Procedure: _h2d
inline void CopyTask::_h2d(
  SpanTask tgt, size_t offset, void* src, size_t bytes, cudaStream_t stream
) {

  auto& t = tgt._node->_span_handle();
  
  auto ptr = static_cast<char*>(t.d_data) + offset;

  HF_CHECK_CUDA(
    cudaMemcpyAsync(
      ptr, src, bytes, cudaMemcpyHostToDevice, stream
    ),
    "copy (H2D) task '", name(), "' failed\n",
    "target span/size/offset: ", t.d_data, '/', t.d_size, '/', offset, '\n',
    "source host/size: ", src, '/', bytes
  );
}

// Procedure: _d2d
inline void CopyTask::_d2d(
  SpanTask tgt, size_t t_offset, 
  SpanTask src, size_t s_offset, 
  size_t bytes,
  cudaStream_t stream
) {
    
  auto& t = tgt._node->_span_handle();
  auto& s = src._node->_span_handle();
    
  auto tptr = static_cast<char*>(t.d_data) + t_offset;
  auto sptr = static_cast<char*>(s.d_data) + s_offset;
    
  HF_CHECK_CUDA(
    cudaMemcpyAsync(
      tptr, sptr, bytes, cudaMemcpyDeviceToDevice, stream
    ),
    "copy (D2D) task '", name(), "' failed\n",
    "target span/size/offset: ", t.d_data, '/', t.d_size, '/', t_offset, '\n',
    "source span/size/offset: ", s.d_data, '/', s.d_size, '/', s_offset, '\n',
    "bytes to copy: ", bytes
  );
}*/
