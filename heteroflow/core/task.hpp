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
    
    TaskBase() = default;
    TaskBase(Node*);
    TaskBase(const TaskBase&) = default;

    TaskBase& operator = (const TaskBase&) = default;

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
  friend class TaskBase<SpanTask>;
  friend class TaskBase<CopyTask>;
  friend class TaskBase<KernelTask>;
  friend class TaskBase<FillTask>;

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
    HostTask host(C&& callable);
    
  private:

    HostTask(Node*);
};

inline HostTask::HostTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

template <typename C>
HostTask HostTask::host(C&& callable) {
  HF_THROW_IF(!_node, "host task is empty");
  _node->_host_handle().work = std::forward<C>(callable);
  return *this;
}

// ----------------------------------------------------------------------------

/**
@class SpanTask

@brief the handle to a span task
*/
class SpanTask : public TaskBase<SpanTask> {
  
  friend class TaskBase<HostTask>;
  friend class TaskBase<SpanTask>;
  friend class TaskBase<CopyTask>;
  friend class TaskBase<KernelTask>;
  friend class TaskBase<FillTask>;
  
  friend class KernelTask;
  friend class CopyTask;
  friend class FillTask;

  friend class FlowBuilder;
  
  friend PointerCaster to_kernel_argument(SpanTask);
  
  using node_handle_t = Node::Span;

  public:

    /**
    @brief constructs an empty span task handle
    */
    SpanTask() = default;
    
    /**
    @brief copy constructor
    */
    SpanTask(const SpanTask&) = default;

    /**
    @brief copy assignment
    */
    SpanTask& operator = (const SpanTask&) = default;
    
    /**
    @brief creates a memory span on GPU
    */
    template <typename N>
    SpanTask span(N&& bytes);

    /**
    @brief creates a memory span on GPU and copies data from a host memory
    */
    template <typename P, typename N>
    SpanTask span(P&& ptr, N&& bytes);

  private:
    
    SpanTask(Node*);
    
    void* _d_data();

    void _allocate(cuda::Allocator&, size_t);
};

// Constructor
inline SpanTask::SpanTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Function: _d_data
inline void* SpanTask::_d_data() {
  return _node->_span_handle().d_data;
}

// Procedure: _allocate
inline void SpanTask::_allocate(cuda::Allocator& a, size_t N) {
  auto& h = _node->_span_handle();
  if(h.d_size < N) {
    h.d_size = N;
    a.deallocate(h.d_data);
    h.d_data = a.allocate(h.d_size);
  }
}
   
// Function: span
template<typename N>
SpanTask SpanTask::span(N&& bytes) {
   
  _node->_span_handle().work = [
    p=*this, bytes=std::forward<N>(bytes)
  ] (cuda::Allocator& a, cudaStream_t) mutable {
    p._allocate(a, bytes);
  };

  return *this;
}

// Function: span
template<typename P, typename N>
SpanTask SpanTask::span(P&& data, N&& bytes) {
   
  _node->_span_handle().work = [
    task=*this, data=std::forward<P>(data), bytes=std::forward<N>(bytes)
  ] (cuda::Allocator& a, cudaStream_t stream) mutable {

    task._allocate(a, bytes);
    
    auto& s = task._node->_span_handle();

    HF_CHECK_CUDA(
      cudaMemcpyAsync(
        s.d_data, data, bytes, cudaMemcpyHostToDevice, stream
      ),
      "span task '", task.name(), "' failed on H2D transfers\n",
      "source span/size: ", s.d_data, '/', s.d_size, '\n',
      "target host/size: ", data, '/', bytes
    );
  };

  return *this;
}

// ----------------------------------------------------------------------------

/**
@class CopyTask

@brief the handle to a copy task
*/
class CopyTask : public TaskBase<CopyTask> {
  
  friend class TaskBase<HostTask>;
  friend class TaskBase<SpanTask>;
  friend class TaskBase<CopyTask>;
  friend class TaskBase<KernelTask>;
  friend class TaskBase<FillTask>;

  friend class FlowBuilder;
  
  using node_handle_t = Node::Copy;

  public:

    /**
    @brief constructs an empty copy task handle
    */
    CopyTask() = default;
    
    /**
    @brief copy constructor
    */
    CopyTask(const CopyTask&) = default;

    /**
    @brief copy assignment
    */
    CopyTask& operator = (const CopyTask&) = default;
    
    /**
    @brief performs device-to-host data transfers
    */
    template <typename P, typename N>
    CopyTask copy(P&& t, SpanTask s, N&& bytes);
   
    /**
    @brief performs device-to-host data transfers
    */
    template <typename P, typename O, typename N>
    CopyTask copy(P&& t, SpanTask s, O&& offset, N&& bytes);

    /**
    @brief performs host-to-device data transfers
    */
    template <typename P, typename N>
    CopyTask copy(SpanTask t, P&& ptr, N&& bytes);
    
    /**
    @brief performs host-to-device data transfers
    */
    template <typename O, typename P, typename N>
    CopyTask copy(SpanTask t, O&& offset, P&& ptr, N&& bytes);

    /**
    @brief performs device-to-device data transfers
    */
    template <typename N>
    CopyTask copy(SpanTask t, SpanTask s, N&& bytes);
    
    /**
    @brief performs device-to-device data transfers
    */
    template <typename OS, typename N>
    CopyTask copy(SpanTask t, SpanTask s, OS&& s_offset, N&& bytes);
    
    /**
    @brief performs device-to-device data transfers
    */
    template <typename OT, typename N>
    CopyTask copy(SpanTask t, OT&& t_offset, SpanTask s, N&& bytes);

    /**
    @brief performs device-to-device data transfers
    */
    template <typename OT, typename OS, typename N>
    CopyTask copy(SpanTask t, OT&& t_offset, SpanTask s, OS&& s_offset, N&& bytes);

  private:

    CopyTask(Node* node);

    void _h2d(SpanTask, size_t, void*, size_t, cudaStream_t);
    void _d2h(void*, SpanTask, size_t, size_t, cudaStream_t);
    void _d2d(SpanTask, size_t, SpanTask, size_t, size_t, cudaStream_t stream);
};

inline CopyTask::CopyTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Function: copy (D2H)
template <typename P, typename N>
CopyTask CopyTask::copy(P&& tgt, SpanTask src, N&& n) {

  HF_THROW_IF(!_node,  "copy task can't be empty");
  HF_THROW_IF(!src, "source span task can't be empty");

  auto& h = _node->_copy_handle();

  h.span = src._node;
  h.direction = cudaMemcpyDeviceToHost;

  h.work = [
    task  = *this, 
    tgt   = std::forward<P>(tgt), 
    src   = src,
    bytes = std::forward<N>(n)
  ] (cudaStream_t stream) mutable {
    task._d2h(tgt, src, 0, bytes, stream);
  };

  return *this;
}

// Function: copy (D2H)
template <typename P, typename O, typename N>
CopyTask CopyTask::copy(P&& tgt, SpanTask src, O&& o, N&& n) {

  HF_THROW_IF(!_node, "copy task can't be empty");
  HF_THROW_IF(!src, "source span task can't be empty");
  
  auto& h = _node->_copy_handle();

  h.span = src._node;
  h.direction = cudaMemcpyDeviceToHost;
  
  h.work = [
    task   = *this, 
    tgt    = std::forward<P>(tgt), 
    src    = src,
    offset = std::forward<O>(o),
    bytes  = std::forward<N>(n)
  ] (cudaStream_t stream) mutable {
    task._d2h(tgt, src, offset, bytes, stream);
  };

  return *this;
}

// Function: copy (H2D)
template <typename P, typename N>
CopyTask CopyTask::copy(SpanTask tgt, P&& src, N&& bytes) {

  HF_THROW_IF(!_node, "copy task can't be empty");
  HF_THROW_IF(!tgt, "target span task can't be empty");
  
  auto& h = _node->_copy_handle();

  h.span = tgt._node;
  h.direction = cudaMemcpyHostToDevice;

  h.work = [
    task  = *this, 
    tgt   = tgt, 
    src   = std::forward<P>(src),
    bytes = std::forward<N>(bytes)
  ] (cudaStream_t stream) mutable {
    task._h2d(tgt, 0, src, bytes, stream);
  };

  return *this;
}

// Function: copy (H2D)
template <typename O, typename P, typename N>
CopyTask CopyTask::copy(SpanTask tgt, O&& offset, P&& src, N&& bytes) {

  HF_THROW_IF(!_node, "copy task can't be empty");
  HF_THROW_IF(!tgt, "target span task can't be empty");
  
  auto& h = _node->_copy_handle();

  h.span = tgt._node;
  h.direction = cudaMemcpyHostToDevice;

  h.work = [
    task   = *this, 
    tgt    = tgt, 
    offset = std::forward<O>(offset),
    src    = std::forward<P>(src),
    bytes  = std::forward<N>(bytes)
  ] (cudaStream_t stream) mutable {
    task._h2d(tgt, offset, src, bytes, stream);
  };

  return *this;
}

// Function: copy (D2D)
template <typename N>
CopyTask CopyTask::copy(SpanTask tgt, SpanTask src, N&& bytes) {

  HF_THROW_IF(!_node, "copy task can't be empty");
  HF_THROW_IF(!tgt, "target span task can't be empty");
  HF_THROW_IF(!src, "source span task can't be empty");
  
  auto& h = _node->_copy_handle();

  h.span = src._node;
  h.direction = cudaMemcpyDeviceToDevice;

  h.work = [
    task  = *this, 
    tgt   = tgt, 
    src   = src,
    bytes = std::forward<N>(bytes)
  ] (cudaStream_t stream) mutable {
    task._d2d(tgt, 0, src, 0, bytes, stream);
  };

  return *this;
}

// Function: copy (D2D)
template <typename OS, typename N>
CopyTask CopyTask::copy(SpanTask t, SpanTask s, OS&& s_offset, N&& bytes) {
  
  HF_THROW_IF(!_node, "copy task can't be empty");
  HF_THROW_IF(!t, "target span task can't be empty");
  HF_THROW_IF(!s, "source span task can't be empty");
  
  auto& h = _node->_copy_handle();

  h.span = s._node;
  h.direction = cudaMemcpyDeviceToDevice;

  h.work = [
    task     = *this, 
    tgt      = t, 
    src      = s,
    s_offset = std::forward<OS>(s_offset),
    bytes    = std::forward<N>(bytes)
  ] (cudaStream_t stream) mutable {
    task._d2d(tgt, 0, src, s_offset, bytes, stream);
  };

  return *this;
}

// Function: copy (D2D)
template <typename OT, typename N>
CopyTask CopyTask::copy(
  SpanTask tgt, OT&& tgt_offset, SpanTask src, N&& bytes
) {

  HF_THROW_IF(!_node, "copy task can't be empty");
  HF_THROW_IF(!tgt, "target span task can't be empty");
  HF_THROW_IF(!src, "source span task can't be empty");
  
  auto& h = _node->_copy_handle();

  h.span = src._node;
  h.direction = cudaMemcpyDeviceToDevice;

  h.work = [
    task     = *this, 
    tgt      = tgt, 
    t_offset = std::forward<OT>(tgt_offset),
    src      = src,
    bytes    = std::forward<N>(bytes)
  ] (cudaStream_t stream) mutable {
    task._d2d(tgt, t_offset, src, 0, bytes, stream);
  };

  return *this;
}

// Function: copy (D2D)
template <typename OT, typename OS, typename N>
CopyTask CopyTask::copy(
  SpanTask tgt, OT&& tgt_offset, SpanTask src, OS&& src_offset, N&& bytes
) {

  HF_THROW_IF(!_node, "copy task can't be empty");
  HF_THROW_IF(!tgt, "target span task can't be empty");
  HF_THROW_IF(!src, "source span task can't be empty");
  
  auto& h = _node->_copy_handle();

  h.span = src._node;
  h.direction = cudaMemcpyDeviceToDevice;

  h.work = [
    task     = *this, 
    tgt      = tgt, 
    t_offset = std::forward<OT>(tgt_offset),
    src      = src,
    s_offset = std::forward<OS>(src_offset),
    bytes    = std::forward<N>(bytes)
  ] (cudaStream_t stream) mutable {
    task._d2d(tgt, t_offset, src, s_offset, bytes, stream);
  };

  return *this;
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

// Procedure: _d2h
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
}

// ----------------------------------------------------------------------------

/**
@class FillTask
@brief handle of a fill task on two GPU memory
*/
class FillTask : public TaskBase<FillTask> {

  friend class TaskBase<HostTask>;
  friend class TaskBase<SpanTask>;
  friend class TaskBase<CopyTask>;
  friend class TaskBase<FillTask>;
  friend class TaskBase<KernelTask>;

  friend class FlowBuilder;
  
  using node_handle_t = Node::Copy;

  public:

    /**
    @brief constructs an empty copy task handle
    */
    FillTask() = default;
    
    /**
    @brief copy constructor
    */
    FillTask(const FillTask&) = default;

    /**
    @brief copy assignment
    */
    FillTask& operator = (const FillTask&) = default;
    
    /**
    @brief fills each byte in a span with a value
    */
    template <typename N, typename V>
    FillTask fill(SpanTask span, N&& bytes, V&& value);
    
    /**
    @brief fills each byte in a span with a value
    */
    template <typename O, typename N, typename V>
    FillTask fill(SpanTask span, O&& offset, N&& bytes, V&& value);
    
  private:

    FillTask(Node* node);

    void _fill(SpanTask, size_t, size_t, int, cudaStream_t);
};

// Constructor
inline FillTask::FillTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Function: fill
template <typename N, typename V>
FillTask FillTask::fill(SpanTask span, N&& bytes, V&& value) {
  
  HF_THROW_IF(!_node, "fill task can't be empty");
  HF_THROW_IF(!span, "target span task can't be empty");
  
  auto& h = _node->_fill_handle();

  h.span = span._node;

  h.work = [
    task  = *this, 
    span  = span,
    bytes = std::forward<N>(bytes),
    value = std::forward<V>(value)
  ] (cudaStream_t stream) mutable {
    task._fill(span, 0, bytes, value, stream);
  };

  return *this;
}

template <typename O, typename N, typename V>
FillTask FillTask::fill(SpanTask span, O&& offset, N&& bytes, V&& value) {
  
  HF_THROW_IF(!_node, "fill task can't be empty");
  HF_THROW_IF(!span, "target span task can't be empty");
  
  auto& h = _node->_fill_handle();

  h.span = span._node;

  h.work = [
    task  = *this, 
    span  = span,
    offset= std::forward<O>(offset),
    bytes = std::forward<N>(bytes),
    value = std::forward<V>(value)
  ] (cudaStream_t stream) mutable {
    task._fill(span, offset, bytes, value, stream);
  };

  return *this;
}

// Procedure: _fill
inline void FillTask::_fill(
  SpanTask span, size_t offset, size_t bytes, int v, cudaStream_t stream
) {

  auto& h = span._node->_span_handle();

  auto ptr = static_cast<char*>(h.d_data) + offset;

  HF_CHECK_CUDA(
    cudaMemsetAsync(ptr, v, bytes, stream),
    "fill task '", name(), "' failed\n",
    "target span/size/offset: ", h.d_data, '/', h.d_size, '/', offset, '\n',
    "filled value/bytes: ", v, '/', bytes
  );
}

// ----------------------------------------------------------------------------

/**
@class KernelTask

@brief the handle to a kernel task
*/
class KernelTask : public TaskBase<KernelTask> {
  
  friend class TaskBase<HostTask>;
  friend class TaskBase<SpanTask>;
  friend class TaskBase<CopyTask>;
  friend class TaskBase<FillTask>;
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
    @brief offloads computations onto a kernel

    @tparam G grid type
    @tparam B block type
    @tparam S shared memory size type
    @tparam F kernel function type
    @tparam ArgsT... kernel function argument types
    
    @param grid arguments to construct the grid of type dim3
    @param block arguments to construct the block of type dim3
    @param shm argument to construct the shared memory size of type size_t
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
    
    PointerCaster _to_argument(SpanTask);
    
    void _gather_sources(void);
    void _gather_sources(SpanTask);
    
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
inline PointerCaster KernelTask::_to_argument(SpanTask task) { 
  HF_THROW_IF(!task, "span task is empty");
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
void KernelTask::_gather_sources(SpanTask task) {
  HF_THROW_IF(!_node, "kernel task cannot operate on empty span task");
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

	HF_CHECK_CUDA(cudaPeekAtLastError(), 
    "failed to launch kernel task '", name(), "'\n",
    "grid=", g.x, 'x', g.y, 'x', g.z, '\n',
    "block=", b.x, 'x', b.y, 'x', b.z, '\n',
    "shm=", s
  );
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
  
  static_assert(
    function_traits<F>::arity == sizeof...(args), 
    "argument arity mismatches"
  );
  
  HF_THROW_IF(!_node, "kernel task is empty");
  
  auto& h = _node->_kernel_handle();
  
  // clear the source span tasks
  h.sources.clear();
  
  // extract source span tasks
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



