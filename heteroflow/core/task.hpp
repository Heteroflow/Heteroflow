#pragma once

#include "graph.hpp"

namespace hf {

/**
@class TaskBase

@brief the base from which all task handles are derived

The class defines a set of common methods used by all types of handles
such as adding precedence links, querying data members, 
changing the name, and so on.

Users are not allowed to directly control such class.
*/
template <typename Derived>
class TaskBase {

  public:
    
    /**
    @brief copy constructor
    */
    TaskBase(const TaskBase&) = default;

    /**
    @brief copy assignment
    */
    TaskBase& operator = (const TaskBase&) = default;

    /**
    @brief queries the name of the task
    */
    const std::string& name() const;
    
    /**
    @brief queries the number of successors of the task
    */
    size_t num_successors() const;

    /**
    @brief queries the number of predecessors of the task
    */
    size_t num_dependents() const;
    
    /**
    @brief queries if the task handle is empty (true for empty)
    */
    bool empty() const;

    /**
    @brief queries if the task handle is empty (true for non-empty)
    */
    operator bool () const;

    /**
    @brief adds precedence links from this task to others

    @tparam Ts... task parameter pack

    @param tasks tasks to precede
    
    @return task handle (derived type)
    */
    template <typename... Ts>
    Derived precede(Ts&&... tasks);
    
    /**
    @brief adds successor links from others to this task

    @tparam Ts... task parameter pack

    @param tasks tasks to succeed
    
    @return task handle (derived type)
    */
    template <typename... Ts>
    Derived succeed(Ts&&... tasks);

    /**
    @brief assigns a name to the task
    
    @tparam S string type

    @param name a @std_string acceptable string
    
    @return task handle (derived type)
    */
    template <typename S>
    Derived name(S&& name);

  protected:
    
    TaskBase(Node*);

    Node* _node {nullptr};
   
    template <typename T>
    void _precede(T&&);
    
    template <typename T, typename... Rest>
    void _precede(T&&, Rest&&...);
    
    template <typename T>
    void _succeed(T&&);
    
    template <typename T, typename... Rest>
    void _succeed(T&&, Rest&&...);
};

// Procedure
template <typename Derived>
inline TaskBase<Derived>::TaskBase(Node* node) :
  _node {node} {
}

// Function: name
template <typename Derived>
inline const std::string& TaskBase<Derived>::name() const {
  HF_THROW_IF(!_node, "can't query the name of an empty task");
  return _node->_name;
}

// Function: num_successors
template <typename Derived>
inline size_t TaskBase<Derived>::num_successors() const {
  HF_THROW_IF(!_node, "empty task has no successors");
  return _node->_successors.size();
}

// Function: num_dependents
template <typename Derived>
inline size_t TaskBase<Derived>::num_dependents() const {
  HF_THROW_IF(!_node, "empty task has no dependents");
  return _node->_dependents.size();
}

// Procedure: precede
template <typename Derived>
template <typename T>
void TaskBase<Derived>::_precede(T&& other) {
  _node->_precede(other._node);
}

// Procedure: _precede
template <typename Derived>
template <typename T, typename... Ts>
void TaskBase<Derived>::_precede(T&& task, Ts&&... others) {
  _precede(std::forward<T>(task));
  _precede(std::forward<Ts>(others)...);
}

// Procedure: succeed
template <typename Derived>
template <typename T>
void TaskBase<Derived>::_succeed(T&& other) {
  other._node->_precede(_node);
}

// Procedure: _succeed
template <typename Derived>
template <typename T, typename... Ts>
void TaskBase<Derived>::_succeed(T&& task, Ts&&... others) {
  _succeed(std::forward<T>(task));
  _succeed(std::forward<Ts>(others)...);
}

// Function: succeed
template <typename Derived>
template <typename... Ts>
Derived TaskBase<Derived>::succeed(Ts&&... tasks) {
  HF_THROW_IF(!_node, "task is empty");
  _succeed(std::forward<Ts>(tasks)...);
  return Derived(_node);
}

// Function: empty
template <typename Derived>
inline bool TaskBase<Derived>::empty() const {
  return _node == nullptr;
}

// Operator
template <typename Derived>
inline TaskBase<Derived>::operator bool() const {
  return _node != nullptr;
}

// Procedure: name
template <typename Derived>
template <typename S>
Derived TaskBase<Derived>::name(S&& name) {
  HF_THROW_IF(!_node, "task is empty");
  _node->_name = std::forward<S>(name);
  return Derived(_node);
}

template <typename Derived>
template <typename... Ts>
Derived TaskBase<Derived>::precede(Ts&&... tasks) {
  HF_THROW_IF(!_node, "task is empty");
  _precede(std::forward<Ts>(tasks)...);
  return Derived(_node);
}

// ----------------------------------------------------------------------------

/**
@class HostTask
@brief the handle to a host (cpu) task
*/
class HostTask : public TaskBase<HostTask> {

  friend class TaskBase<HostTask>;
  friend class TaskBase<PullTask>;
  friend class TaskBase<PushTask>;
  friend class TaskBase<KernelTask>;
  friend class TaskBase<TransferTask>;

  friend class FlowBuilder;

  using node_handle_t = Node::Host;
  
  public:

    /**
    @brief constructs an empty host task handle
    */
    HostTask() = default;
    
    /**
    @brief copy constructor
    */
    HostTask(const HostTask&) = default;

    /**
    @brief copy assignment
    */
    HostTask& operator = (const HostTask&) = default;

    /**
    @brief assigns a work to the host task
    
    @tparam C callable type
    
    @param callable a callable object acceptable to std::function
    */
    template <typename C>
    HostTask work(C&& callable);
    
  private:

    HostTask(Node*);
};

inline HostTask::HostTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

template <typename C>
HostTask HostTask::work(C&& callable) {
  HF_THROW_IF(!_node, "host task is empty");
  _node->_host_handle().work = std::forward<C>(callable);
  return *this;
}

// ----------------------------------------------------------------------------

/**
@class PullTask

@brief the handle to a pull task
*/
class PullTask : public TaskBase<PullTask> {
  
  friend class TaskBase<HostTask>;
  friend class TaskBase<PullTask>;
  friend class TaskBase<PushTask>;
  friend class TaskBase<KernelTask>;
  friend class TaskBase<TransferTask>;
  
  friend class KernelTask;
  friend class PushTask;
  friend class TransferTask;

  friend class FlowBuilder;
  
  friend PointerCaster to_kernel_argument(PullTask);
  
  using node_handle_t = Node::Pull;

  public:

    /**
    @brief constructs an empty pull task handle
    */
    PullTask() = default;
    
    /**
    @brief copy constructor
    */
    PullTask(const PullTask&) = default;

    /**
    @brief copy assignment
    */
    PullTask& operator = (const PullTask&) = default;

    /**
    @brief alters the host memory block to copy to GPU

    @tparam ArgsT argements types
    @param args arguments to forward to construct a span object
    */
    template<typename P, typename N>
    PullTask pull(P&& pointer, N&& bytes);

    template<typename N>
    PullTask pull(std::nullptr_t, N&& bytes);
    
    template<typename N, typename V>
    PullTask pull(std::nullptr_t, N&& bytes, V&& value);
   
  private:
    
    PullTask(Node*);
    
    void* _d_data();
    
    template <typename P>
    void _invoke_pull(P&&, size_t, cuda::Allocator&, cudaStream_t);

    void _invoke_pull(std::nullptr_t, size_t, cuda::Allocator&, cudaStream_t);

    template <typename V>
    void _invoke_pull(std::nullptr_t, size_t, V&&, cuda::Allocator&, cudaStream_t);
};

// Constructor
inline PullTask::PullTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Function: d_data
inline void* PullTask::_d_data() {
  return nonstd::get<node_handle_t>(_node->_handle).d_data;
}

// Function: pull
template<typename P, typename N>
PullTask PullTask::pull(P&& ptr, N&& size) {
   
  _node->_pull_handle().work = [
    p=*this, ptr=std::forward<P>(ptr), size=std::forward<N>(size)
  ] (cuda::Allocator& a, cudaStream_t s) mutable {
    p._invoke_pull(ptr, size, a, s);
  };

  return *this;
}

// Function: pull
template<typename N>
PullTask PullTask::pull(std::nullptr_t, N&& size) {
   
  _node->_pull_handle().work = [
    p=*this, size=std::forward<N>(size)
  ] (cuda::Allocator& a, cudaStream_t s) mutable {
    p._invoke_pull(nullptr, size, a, s);
  };

  return *this;
}

// Function: pull
template<typename N, typename V>
PullTask PullTask::pull(std::nullptr_t, N&& size, V&& value) {
   
  _node->_pull_handle().work = [
    p=*this, size=std::forward<N>(size), value=std::forward<V>(value)
  ] (cuda::Allocator& a, cudaStream_t s) mutable {
    p._invoke_pull(nullptr, size, value, a, s);
  };

  return *this;
}

// Function: _invoke_pull
template <typename P>
void PullTask::_invoke_pull(
  P&& h_data, size_t h_size, cuda::Allocator& a, cudaStream_t s
) {
      
  // pull handle
  auto& h = _node->_pull_handle();

  // allocate the global memory
  if(h.d_data == nullptr) {
    assert(h.d_size == 0);
    h.d_size = h_size;
    h.d_data = a.allocate(h.d_size);
  }
  // Check size first (graph is resuable)
  // reallocate the global memory
  else if(h.d_size < h_size) {
    assert(h.d_data != nullptr);
    h.d_size = h_size;
    a.deallocate(h.d_data);
    h.d_data = a.allocate(h.d_size);
  }

  // transfer the memory
  HF_CHECK_CUDA(
    cudaMemcpyAsync(
      h.d_data, h_data, h_size, cudaMemcpyHostToDevice, s
    ),
    "failed to pull memory in task ", name()
  );
}


// Function: _invoke_pull
inline void PullTask::_invoke_pull(
  std::nullptr_t, size_t h_size, cuda::Allocator& a, cudaStream_t
) {
  // pull handle
  auto& h = _node->_pull_handle();

  // allocate the global memory
  if(h.d_data == nullptr) {
    assert(h.d_size == 0);
    h.d_size = h_size;
    h.d_data = a.allocate(h.d_size);
  }
  // Check size first (graph is resuable)
  // reallocate the global memory
  else if(h.d_size < h_size) {
    assert(h.d_data != nullptr);
    h.d_size = h_size;
    a.deallocate(h.d_data);
    h.d_data = a.allocate(h.d_size);
  }
}

// Function: _invoke_pull
template <typename V>
void PullTask::_invoke_pull(
  std::nullptr_t, size_t h_size, V&& value, cuda::Allocator& a, cudaStream_t s
) {
  // pull handle
  auto& h = _node->_pull_handle();

  // allocate the global memory
  if(h.d_data == nullptr) {
    assert(h.d_size == 0);
    h.d_size = h_size;
    h.d_data = a.allocate(h.d_size);
  }
  // Check size first (graph is resuable)
  // reallocate the global memory
  else if(h.d_size < h_size) {
    assert(h.d_data != nullptr);
    h.d_size = h_size;
    a.deallocate(h.d_data);
    h.d_data = a.allocate(h.d_size);
  }

  // Initialize the memory
  HF_CHECK_CUDA(
    cudaMemsetAsync(
      h.d_data, value, h_size, s
    ),
    "failed to initialize memory in task ", name()
  );
}
  
// ----------------------------------------------------------------------------

/**
@class PushTask

@brief the handle to a push task
*/
class PushTask : public TaskBase<PushTask> {
  
  friend class TaskBase<HostTask>;
  friend class TaskBase<PullTask>;
  friend class TaskBase<PushTask>;
  friend class TaskBase<KernelTask>;
  friend class TaskBase<TransferTask>;

  friend class FlowBuilder;
  
  using node_handle_t = Node::Push;

  public:

    /**
    @brief constructs an empty push task handle
    */
    PushTask() = default;
    
    /**
    @brief copy constructor
    */
    PushTask(const PushTask&) = default;

    /**
    @brief copy assignment
    */
    PushTask& operator = (const PushTask&) = default;
    
    /**
    @brief alters the host memory block to push from gpu
    
    @tparam ArgsT argements types
    @param source a source pull task of a gpu memory block
    @param args arguments to forward to construct a span object
    */
    template <typename P, typename N>
    PushTask push(PullTask, P&&, N&&);
   
    template <typename P, typename N, typename O>
    PushTask push(PullTask, P&& p, N&& n, O&& offset);

  private:

    PushTask(Node* node);
    
    template <typename P>
    void _invoke_push(P&&, size_t, size_t, cudaStream_t);

    template <typename P>
    void _invoke_push(P&&, size_t, cudaStream_t);

};

inline PushTask::PushTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Function: push
template <typename P, typename N>
PushTask PushTask::push(PullTask source, P&& p, N&& n) {
  HF_THROW_IF(!_node,  "push task can't be empty");
  HF_THROW_IF(!source, "pull task can't be empty");

  auto& h = _node->_push_handle();

  h.source = source._node;

  h.work = [
    task = *this, 
    ptr  = std::forward<P>(p), 
    size = std::forward<N>(n)
  ] (cudaStream_t stream) mutable {
    task._invoke_push(ptr, size, stream);
  };

  return *this;
}

// Function: push
template <typename P, typename N, typename O>
PushTask PushTask::push(PullTask source, P&& p, N&& n, O&& offset) {

  HF_THROW_IF(!_node,  "push task can't be empty");
  HF_THROW_IF(!source, "pull task can't be empty");

  auto& h = _node->_push_handle();

  h.source = source._node;

  h.work = [
    task   = *this, 
    ptr    = std::forward<P>(p), 
    size   = std::forward<N>(n), 
    offset = std::forward<O>(offset)
  ] (cudaStream_t stream) mutable {
    task._invoke_push(ptr, size, offset, stream);
  };

  return *this;
}

// Procedure: _invoke_push
template <typename P>
void PushTask::_invoke_push(
  P &&h_data, size_t h_size, size_t offset, cudaStream_t stream
) {
  // get the handle and device memory
  auto& h = _node->_push_handle();
  auto& s = h.source->_pull_handle();

  // std::cout << "push " << h_data << ' ' << h_size << std::endl;

  HF_THROW_IF(s.d_data == nullptr || s.d_size < h_size,
    "invalid memory push from ", h.source->_name, " to ", name()
  ); 

  auto ptr = static_cast<unsigned char*>(s.d_data) + offset;

  HF_CHECK_CUDA(
    cudaMemcpyAsync(
      h_data, ptr, h_size, cudaMemcpyDeviceToHost, stream
    ),
    "failed to push memory in task ", name()
  );
}
 

// Procedure: _invoke_push
template <typename P>
void PushTask::_invoke_push(
  P &&h_data, size_t h_size, cudaStream_t stream
) {
  // get the handle and device memory
  auto& h = _node->_push_handle();
  auto& s = h.source->_pull_handle();

  //std::cout << "push " << h_data << ' ' << h_size << std::endl;

  HF_THROW_IF(s.d_data == nullptr || s.d_size < h_size,
    "invalid memory push from ", h.source->_name, " to ", name()
  ); 

  HF_CHECK_CUDA(
    cudaMemcpyAsync(
      h_data, s.d_data, h_size, cudaMemcpyDeviceToHost, stream
    ),
    "failed to push memory in task ", name()
  );
}

// ----------------------------------------------------------------------------

/**
@class TransferTask
@brief handle of a transfer task on two GPU memory
*/
class TransferTask : public TaskBase<TransferTask> {

  friend class TaskBase<HostTask>;
  friend class TaskBase<PullTask>;
  friend class TaskBase<PushTask>;
  friend class TaskBase<TransferTask>;
  friend class TaskBase<KernelTask>;

  friend class FlowBuilder;
  
  using node_handle_t = Node::Push;

  public:

    /**
    @brief constructs an empty push task handle
    */
    TransferTask() = default;
    
    /**
    @brief copy constructor
    */
    TransferTask(const TransferTask&) = default;

    /**
    @brief copy assignment
    */
    TransferTask& operator = (const TransferTask&) = default;
    
    /**
    @brief alters the host memory block to push from gpu
    
    @tparam ArgsT argements types

    @param source a source pull task of a gpu memory block
    @param target a target pull task of a gpu memory block
    @param OT offset of target memory block
    @param OS offset of source memory block
    @param N size of memory block
    */
    template <typename OT, typename OS, typename N>
    TransferTask transfer(
      PullTask dst, PullTask src, OT&& dst_offset, OS&& src_offset, N&& size
    );

  private:

    TransferTask(Node* node);
 
    void _invoke_transfer(cudaStream_t, size_t, size_t, size_t);

};

// Constructor
inline TransferTask::TransferTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Function: transfer 
template <typename OT, typename OS, typename N>
TransferTask TransferTask::transfer(
  PullTask target, PullTask source, OT&& ot, OS &&os, N&& size
) {

  HF_THROW_IF(!_node,  "transfer task can't be empty");
  HF_THROW_IF(!source, "source task can't be empty");
  HF_THROW_IF(!target, "target task can't be empty");

  auto& h = _node->_transfer_handle();
    
  h.source = source._node;
  h.target = target._node;

  h.work = [
    task = *this, 
    ot   = std::forward<OT>(ot), 
    os   = std::forward<OS>(os), 
    size = std::forward<N>(size)
  ] (cudaStream_t stream) mutable {
    task._invoke_transfer(stream, ot, os, size);
  };

  return *this;
}

// Procedure: _invoke_transfer 
inline void TransferTask::_invoke_transfer(
  cudaStream_t stream, size_t ot, size_t os, size_t size
) {
 
  // get the handle and device memory
  auto& h = _node->_transfer_handle();
  auto& source = h.source->_pull_handle();
  auto& target = h.target->_pull_handle();

  HF_THROW_IF(source.d_data == nullptr || target.d_data == nullptr,
    "invalid memory transfer from ", h.source->_name, " to ", name()
  ); 

  auto from_ptr = static_cast<unsigned char*>(source.d_data) + os;
  auto to_ptr = static_cast<unsigned char*>(target.d_data) + ot;

  HF_CHECK_CUDA(
    cudaMemcpyAsync(
      to_ptr, from_ptr, size, cudaMemcpyDeviceToDevice, stream
    ),
    "failed to transfer memory in task ", name()
  );
}


// ----------------------------------------------------------------------------

/**
@class KernelTask

@brief the handle to a kernel task
*/
class KernelTask : public TaskBase<KernelTask> {
  
  friend class TaskBase<HostTask>;
  friend class TaskBase<PullTask>;
  friend class TaskBase<PushTask>;
  friend class TaskBase<TransferTask>;
  friend class TaskBase<KernelTask>;

  friend class FlowBuilder;
  
  using node_handle_t = Node::Kernel;

  public:

    /**
    @brief constructs an empty kernel task handle
    */
    KernelTask() = default;
    
    /**
    @brief copy constructor
    */
    KernelTask(const KernelTask&) = default;

    /**
    @brief copy assignment
    */
    KernelTask& operator = (const KernelTask&) = default;
    
    /**
    @brief assign a kernel

    @tparam G grid type
    @tparam B block type
    @tparam S shared memory size type
    @tparam F kernel function type
    @tparam ArgsT... kernel function argument types
    
    @param grid argument to construct a grid of type dim3
    @param block argument to construct a block of type dim3
    @param shm argument to construct a shared memory size of type size_t
    @param func kernel function
    @param args... arguments to forward to the kernel function

    @return KernelTask handle

    Kernel task executes a kernel in a given configuration.
    */
    template <typename G, typename B, typename S, typename F, typename... ArgsT>
    KernelTask kernel(
      G&& grid, B&& block, S&& shm, F&& func, ArgsT&&... args
    );
    
  private:

    KernelTask(Node* node);

    template <typename T>
    auto _to_argument(T&& t);
    
    PointerCaster _to_argument(PullTask);
    
    void _gather_sources(void);
    void _gather_sources(PullTask);
    
    template <typename T>
    void _gather_sources(T&&);

    template <typename T, typename... Ts>
    void _gather_sources(T&&, Ts&&...);

    template<typename K, typename T, size_t ... I>
    void _invoke_kernel(
      dim3, dim3, size_t, cudaStream_t, K, T, std::index_sequence<I ...>
    );

    template<typename K, typename T>
    void _invoke_kernel(dim3, dim3, size_t, cudaStream_t, K, T);
};

// Constructor
inline KernelTask::KernelTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Function: _to_argument
template <typename T>
auto KernelTask::_to_argument(T&& t) { 
  return std::forward<T>(t); 
}

// Function: _to_argument
inline PointerCaster KernelTask::_to_argument(PullTask task) { 
  HF_THROW_IF(!task, "pull task is empty");
  return PointerCaster{task._d_data()}; 
}

// Procedure: _gather_sources
inline void KernelTask::_gather_sources(void) {
}
    
// Procedure: _gather_sources
template <typename T>
void KernelTask::_gather_sources(T&&) {
}

// Procedure: _gather_sources
void KernelTask::_gather_sources(PullTask task) {
  HF_THROW_IF(!_node, "kernel task cannot operate on empty pull task");
  _node->_kernel_handle().sources.push_back(task._node);
}
    
// Procedure: _gather_sources
template <typename T, typename... Ts>
void KernelTask::_gather_sources(T&& task, Ts&&... others) {
  _gather_sources(std::forward<T>(task));
  _gather_sources(std::forward<Ts>(others)...);
}
    
// Procedure: _invoke_kernel
template<typename K, typename T, size_t ... I>
void KernelTask::_invoke_kernel(
  dim3 g, 
  dim3 b, 
  size_t s, 
  cudaStream_t stream, 
  K f, 
  T t, 
  std::index_sequence<I ...>
) {
  auto& h = _node->_kernel_handle();
  f<<<g, b, s, stream>>>(_to_argument(std::get<I>(t))...);
}

// Procedure: _invoke_kernel
template<typename K, typename T>
void KernelTask::_invoke_kernel(
  dim3 g,
  dim3 b,
  size_t s,
  cudaStream_t stream, 
  K f, 
  T t
) {
  static constexpr auto size = std::tuple_size<T>::value;
  _invoke_kernel(g, b, s, stream, f, t, std::make_index_sequence<size>{});
}

// Function: kernel
template <typename G, typename B, typename S, typename F, typename... ArgsT>
KernelTask KernelTask::kernel(
  G&& g, B&& b, S&& s, F&& f, ArgsT&&... args
) {
  
  HF_THROW_IF(!_node, "kernel task is empty");
  
  auto& h = _node->_kernel_handle();
  
  // clear the source pull tasks
  h.sources.clear();
  
  // extract source pull tasks
  _gather_sources(args...);
  
  // assign kernel work. here we create a task to avoid dangling ref
  // due to the builder pattern we enabled here
  h.work = [
    k=*this, 
    g=std::forward<G>(g),
    b=std::forward<B>(b),
    s=std::forward<S>(s),
    f=std::forward<F>(f), 
    t=std::make_tuple(std::forward<ArgsT>(args)...)
  ] (cudaStream_t stream) mutable {
    k._invoke_kernel(g, b, s, stream, f, t);    
  };

  return *this;
}

}  // end of namespace hf -----------------------------------------------------



