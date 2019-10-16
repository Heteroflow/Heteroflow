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
    
    template<typename T, size_t ... I>
    auto _make_span(T, std::index_sequence<I ...>);

    template <typename T>
    auto _make_span(T);
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

// Procedure: _invoke_kernel
template <typename Derived>
template <typename T, size_t ... I>
auto TaskBase<Derived>::_make_span(T t, std::index_sequence<I ...>) {
  return hf::make_span(std::get<I>(t)...);
}

// Procedure: _make_span
template <typename Derived>
template <typename T>
auto TaskBase<Derived>::_make_span(T t) {
  static constexpr auto size = std::tuple_size<T>::value;
  return _make_span(t, std::make_index_sequence<size>{});
}

// ----------------------------------------------------------------------------

/**
@class PullTask

@brief the handle to a host (cpu) task
*/
class HostTask : public TaskBase<HostTask> {
  
  friend class TaskBase<HostTask>;
  friend class TaskBase<PullTask>;
  friend class TaskBase<PushTask>;
  friend class TaskBase<KernelTask>;

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
  
  friend class KernelTask;
  friend class PushTask;

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
    template <typename... ArgsT>
    PullTask pull(ArgsT&&... args);
    
  private:
    
    PullTask(Node*);
    
    void* _d_data();
    
    template <typename T>
    void _invoke_pull(T, cuda::Allocator&, cudaStream_t);
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
template <typename... ArgsT>
PullTask PullTask::pull(ArgsT&&... args) {
   
  _node->_pull_handle().work = [
    p=*this, t=make_stateful_tuple(std::forward<ArgsT>(args)...)
  ] (cuda::Allocator& a, cudaStream_t s) mutable {
    p._invoke_pull(t, a, s);
  };

  return *this;
}
    
// Function: _invoke_pull
template <typename T>
void PullTask::_invoke_pull(
  T t, cuda::Allocator& a, cudaStream_t s
) {
      
  // obtain the data span
  auto h_span = _make_span(t);
  auto h_data = h_span.data();        // can't be nullptr
  auto h_size = h_span.size_bytes();
  
  // pull handle
  auto& h = _node->_pull_handle();

  std::cout << "pull " << h_data << ' ' << h_size << std::endl;

  // allocate the global memory
  if(h.d_data == nullptr) {
    assert(h.d_size == 0);
    h.d_size = h_size;
  }
  // reallocate the global memory
  else if(h.d_size < h_size) {
    assert(h.d_data != nullptr);
    h.d_size = h_size;
    a.deallocate(h.d_data);
  }

  h.d_data = a.allocate(h.d_size);

  //std::cout << "global memory " << h.d_data << ' ' << h.d_size << std::endl;
  
  // transfer the memory
  HF_CHECK_CUDA(
    cudaMemcpyAsync(
      h.d_data, h_data, h_size, cudaMemcpyHostToDevice, s
    ),
    "failed to pull memory in task ", name()
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
    template <typename... ArgsT>
    PushTask push(PullTask source, ArgsT&&... args);
    
  private:

    PushTask(Node* node);
    
    template <typename T>
    void _invoke_push(T, cudaStream_t);
};

inline PushTask::PushTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Function: push
template <typename... ArgsT>
PushTask PushTask::push(PullTask source, ArgsT&&... args) {

  HF_THROW_IF(!_node,  "push task can't be empty");
  HF_THROW_IF(!source, "pull task can't be empty");

  auto& h = _node->_push_handle();
    
  h.source = source._node;
  h.work = [
    p=*this, t=make_stateful_tuple(std::forward<ArgsT>(args)...)
  ] (cudaStream_t stream) mutable {
    p._invoke_push(t, stream);
  };

  return *this;
}

// Procedure: _invoke_push
template <typename T>
void PushTask::_invoke_push(T t, cudaStream_t stream) {
 
  // obtain the data span
  auto h_span = _make_span(t);
  auto h_data = h_span.data();        // can't be nullptr
  auto h_size = h_span.size_bytes();
  
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
@class KernelTask

@brief the handle to a kernel task
*/
class KernelTask : public TaskBase<KernelTask> {
  
  friend class TaskBase<HostTask>;
  friend class TaskBase<PullTask>;
  friend class TaskBase<PushTask>;
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
    @brief alters the x dimension of the grid
    
    @return task handle
    */
    KernelTask grid_x(size_t x);
    
    /**
    @brief alters the y dimension of the grid
    
    @return task handle
    */
    KernelTask grid_y(size_t y);
    
    /**
    @brief alters the z dimension of the grid
    
    @return task handle
    */
    KernelTask grid_z(size_t z);
    
    /**
    @brief alters the grid dimension in the form of {x, y, z}
    
    @return task handle
    */
    KernelTask grid(const ::dim3& grid);

    /**
    @brief queries the x dimension of the grid
    */
    size_t grid_x() const;
    
    /**
    @brief queries the y dimension of the grid
    */
    size_t grid_y() const;
    
    /**
    @brief queries the z dimension of the grid
    */
    size_t grid_z() const;

    /**
    @brief query the grid dimension
    */
    const ::dim3& grid() const;
    
    /**
    @brief alters the x dimension of the block
    
    @return task handle
    */
    KernelTask block_x(size_t x);
    
    /**
    @brief alters the y dimension of the block
    
    @return task handle
    */
    KernelTask block_y(size_t y);
    
    /**
    @brief alters the z dimension of the block
    
    @return task handle
    */
    KernelTask block_z(size_t z);
    
    /**
    @brief alters the block dimension in the form of {x, y, z}
    
    @return task handle
    */
    KernelTask block(const ::dim3& block);
    
    /**
    @brief alters the shared memory size 

    @return task handle
    */
    KernelTask shm(size_t);

    /**
    @brief queries the x dimension of the block
    */
    size_t block_x() const;
    
    /**
    @brief queries the y dimension of the block
    */
    size_t block_y() const;
    
    /**
    @brief queries the z dimension of the block
    */
    size_t block_z() const;

    /**
    @brief query the block dimension
    */
    const ::dim3& block() const;
    
    /**
    @brief query the shared memory size
    */
    size_t shm() const;
    
    /**
    @brief assign a kernel 

    @tparam ArgsT... argument types

    @param func kernel function
    @param args... arguments to forward to the kernel function

    @return KernelTask handle

    The function performs default configuration to launch the kernel.
    */
    template <typename F, typename... ArgsT>
    KernelTask kernel(F&& func, ArgsT&&... args);
    
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
    void _invoke_kernel(cudaStream_t, K, T, std::index_sequence<I ...>);

    template<typename K, typename T>
    void _invoke_kernel(cudaStream_t, K, T);
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
void KernelTask::_gather_sources(T&& other) {
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
  cudaStream_t s, K f, T t, std::index_sequence<I ...>
) {
  auto& h = _node->_kernel_handle();
  f<<<h.grid, h.block, h.shm, s>>>(_to_argument(std::get<I>(t))...);
}

// Procedure: _invoke_kernel
template<typename K, typename T>
void KernelTask::_invoke_kernel(cudaStream_t s, K f, T t) {
  static constexpr auto size = std::tuple_size<T>::value;
  _invoke_kernel(s, f, t, std::make_index_sequence<size>{});
}

// Procedure: grid_x
inline KernelTask KernelTask::grid_x(size_t x) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).grid.x = x;
  return *this;
}

// Procedure: grid_y
inline KernelTask KernelTask::grid_y(size_t y) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).grid.y = y;
  return *this;
}

// Procedure: grid_z
inline KernelTask KernelTask::grid_z(size_t z) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).grid.z = z;
  return *this;
}

// Procedure: grid
inline KernelTask KernelTask::grid(const ::dim3& grid) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).grid = grid;
  return *this;
}

// Function: grid_x
inline size_t KernelTask::grid_x() const {
  HF_THROW_IF(!_node, "kernel task is empty");
  return nonstd::get<node_handle_t>(_node->_handle).grid.x;
}

// Function: grid_y
inline size_t KernelTask::grid_y() const {
  HF_THROW_IF(!_node, "kernel task is empty");
  return nonstd::get<node_handle_t>(_node->_handle).grid.y;
}

// Function: grid_z
inline size_t KernelTask::grid_z() const {
  HF_THROW_IF(!_node, "kernel task is empty");
  return nonstd::get<node_handle_t>(_node->_handle).grid.z;
}

// Function: grid
inline const ::dim3& KernelTask::grid() const {
  HF_THROW_IF(!_node, "kernel task is empty");
  return nonstd::get<node_handle_t>(_node->_handle).grid;
}

// Procedure: block_x
inline KernelTask KernelTask::block_x(size_t x) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).block.x = x;
  return *this;
}

// Procedure: block_y
inline KernelTask KernelTask::block_y(size_t y) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).block.y = y;
  return *this;
}

// Procedure: block_z
inline KernelTask KernelTask::block_z(size_t z) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).block.z = z;
  return *this;
}

// Procedure: block
inline KernelTask KernelTask::block(const ::dim3& block) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).block = block;
  return *this;
}

// Function: block_x
inline size_t KernelTask::block_x() const {
  HF_THROW_IF(!_node, "kernel task is empty");
  return nonstd::get<node_handle_t>(_node->_handle).block.x;
}

// Function: block_y
inline size_t KernelTask::block_y() const {
  HF_THROW_IF(!_node, "kernel task is empty");
  return nonstd::get<node_handle_t>(_node->_handle).block.y;
}

// Function: block_z
inline size_t KernelTask::block_z() const {
  HF_THROW_IF(!_node, "kernel task is empty");
  return nonstd::get<node_handle_t>(_node->_handle).block.z;
}

// Function: block
inline const ::dim3& KernelTask::block() const {
  HF_THROW_IF(!_node, "kernel task is empty");
  return nonstd::get<node_handle_t>(_node->_handle).block;
}

// Procedure: shm
inline KernelTask KernelTask::shm(size_t sz) {
  _node->_kernel_handle().shm = sz;
  return *this;
}

// Function: shm
inline size_t KernelTask::shm() const {
  HF_THROW_IF(!_node, "kernel task is empty");
  return _node->_kernel_handle().shm;
}

template <typename F, typename... ArgsT>
KernelTask KernelTask::kernel(F&& f, ArgsT&&... args) {
  
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
    f=std::forward<F>(f), 
    t=make_stateful_tuple(std::forward<ArgsT>(args)...)
  ] (cudaStream_t stream) mutable {
    k._invoke_kernel(stream, f, t);    
  };

  return *this;
}

// ----------------------------------------------------------------------------

//template <typename T>
//auto to_kernel_argument(T&& t) -> decltype(std::forward<T>(t)) { 
//  return std::forward<T>(t); 
//}
//
//inline PointerCaster to_kernel_argument(PullTask task) { 
//  HF_THROW_IF(!task, "pull task is empty");
//  return PointerCaster{task._d_data()}; 
//}



}  // end of namespace hf -----------------------------------------------------






