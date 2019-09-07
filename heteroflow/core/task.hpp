#pragma once

#include "graph.hpp"

namespace hf {

// forward declaration
class PushTask;
class PullTask;
class HostTask;
class KernelTask;

// ----------------------------------------------------------------------------

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
  _node->_work = std::forward<C>(callable);
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

    @param source the pointer to the beginning of the host memory block
    @param N number of bytes to pull

    When source is a null pointer, the pull task performs only global
    memory allocation on the device. 
    */
    PullTask pull(const void* source, size_t N);
    
  private:
    
    PullTask(Node*);
    
    void* _d_data();

    void _make_work();
};

// Constructor
inline PullTask::PullTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Procedure: pull
inline PullTask PullTask::pull(const void* source, size_t N) {

  HF_THROW_IF(!_node, "pull task is empty");

  auto& handle = _node->_pull_handle();
  handle.h_data = source;
  handle.h_size = N;

  // assign the work
  _make_work();

  return *this;
}

// Function: d_data
inline void* PullTask::_d_data() {
  return nonstd::get<node_handle_t>(_node->_handle).d_data;
}
  
// Procedure: _make_work
inline void PullTask::_make_work() {  

  _node->_work = [this, v=_node] () {

    auto& h = v->_pull_handle();

    std::cout << "running task " << v->_name << std::endl;

    HF_WITH_CUDA_DEVICE(h.device) {

      // allocate the global memory
      if(h.d_data == nullptr) {
        assert(h.d_size == 0);
        h.d_size = h.h_size;
      }
      // reallocate the global memory
      else if(h.d_size < h.h_size) {
        assert(h.d_data != nullptr);
        h.d_size = h.h_size;
        HF_CHECK_CUDA(::cudaFree(h.d_data),
          "failed to free global memory in task ", name()
        )
      }

      HF_CHECK_CUDA(::cudaMalloc(&(h.d_data), h.d_size),
        "failed to allocate global memory in task ", name()
      );
      
      // nothing to do if the host is null
      if(h.h_data == nullptr) {
        return;
      }

      // transfer the memory
      HF_CHECK_CUDA(
        ::cudaMemcpyAsync(
          h.d_data, h.h_data, h.h_size, cudaMemcpyHostToDevice, h.stream
        ),
        "failed to pull memory in task ", name()
      );
    }
  };
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
    
    @param target the pointer to the beginning of the host memory block
    @param source the source pull task that stores the gpu memory block
    @param N number of bytes to push
    */
    PushTask push(void* target, PullTask source, size_t N);
    
  private:

    PushTask(Node* node);

    void _make_work();
};

inline PushTask::PushTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Function: push
inline PushTask PushTask::push(void* target, PullTask source, size_t N) {

  HF_THROW_IF(!_node,  "push task can't be empty");
  HF_THROW_IF(!source, "pull task can't be empty");
    
  auto& handle = _node->_push_handle();
  handle.h_data = target;
  handle.source = source._node;
  handle.h_size = N;
  
  _make_work();

  return *this;
}

// Procedure: _make_work
inline void PushTask::_make_work() {

  _node->_work = [this, v=_node] () {

    auto& h = v->_push_handle();
    
    if(h.h_data == nullptr || h.h_size == 0) {
      return;
    }
    
    // get the device memory
    auto& s = h.source->_pull_handle();
    
    HF_WITH_CUDA_DEVICE(s.device) {

      HF_THROW_IF(s.d_data == nullptr || s.d_size < h.h_size,
        "invalid memory push from ", h.source->_name, " to ", name()
      ); 

      HF_CHECK_CUDA(
        ::cudaMemcpyAsync(
          h.h_data, s.d_data, h.h_size, cudaMemcpyDeviceToHost, h.stream
        ),
        "failed to push memory in task ", name()
      );
    }
  };
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
KernelTask KernelTask::kernel(F&& func, ArgsT&&... args) {
  
  HF_THROW_IF(!_node, "kernel task is empty");
  
  // clear the source pull tasks
  _node->_kernel_handle().sources.clear();
  
  // extract source pull tasks
  _gather_sources(args...);

  // assign the work
  _node->_work = [this, v=_node, func, args...] () {
    auto& h = v->_kernel_handle();
    HF_WITH_CUDA_DEVICE(h.device) {
      func<<<h.grid, h.block, h.shm, h.stream>>>(
        _to_argument(args)...
      );
    }
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






