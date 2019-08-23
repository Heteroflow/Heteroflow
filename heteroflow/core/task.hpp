#pragma once

#include "graph.hpp"

namespace hf {

// forward declaration
struct PointerCaster;

/**
@class TaskBase

@brief the base from which all task handles are derived

The class defines a set of common methods used by all types of handles
such as adding precedence links, querying data members, 
changing the name, and so on.

Users are not allowed to directly control such class.
*/
class TaskBase {
  
  friend class HostTask;
  friend class PullTask;
  friend class PushTask;
  friend class KernelTask;
  friend class FlowBuilder;

  public:

    /**
    @brief adds precedence links from this task to others

    @tparam Ts... task parameter pack

    @param tasks one or multiple tasks
    */
    template <typename... Ts>
    void precede(Ts&&... tasks);
    
    /**
    @brief assigns a name to the task
    
    @tparam S string type

    @param name a @std_string acceptable string
    */
    template <typename S>
    void name(S&& name);
    
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
    @brief resets the task handle to point to nothing
    */
    void reset();

    /**
    @brief queries if the task handle is empty (true for empty)
    */
    bool empty() const;

    /**
    @brief queries if the task handle is empty (true for non-empty)
    */
    operator bool () const;
  
  private:
    
    TaskBase(Node*);

    Node* _node {nullptr};
   
    template <typename T>
    void _precede(T&&);
    
    template <typename T, typename... Rest>
    void _precede(T&&, Rest&&...);
};

// Procedure
inline TaskBase::TaskBase(Node* node) :
  _node {node} {
}

// Procedure: name
template <typename S>
void TaskBase::name(S&& name) {
  HF_THROW_IF(!_node, "can't assign name to an empty task");
  _node->_name = std::forward<S>(name);
}

// Function: name
inline const std::string& TaskBase::name() const {
  HF_THROW_IF(!_node, "can't query the name of an empty task");
  return _node->_name;
}

// Function: num_successors
inline size_t TaskBase::num_successors() const {
  HF_THROW_IF(!_node, "empty task has no successors");
  return _node->_successors.size();
}

// Function: num_dependents
inline size_t TaskBase::num_dependents() const {
  HF_THROW_IF(!_node, "empty task has no dependents");
  return _node->_dependents.size();
}

// Procedure: precede
template <typename T>
void TaskBase::_precede(T&& other) {
  _node->_precede(other._node);
}

// Procedure: _precede
template <typename T, typename... Ts>
void TaskBase::_precede(T&& task, Ts&&... others) {
  _precede(std::forward<T>(task));
  _precede(std::forward<Ts>(others)...);
}

// Procedure: _precede
template <typename... Ts>
void TaskBase::precede(Ts&&... tasks) {
  HF_THROW_IF(_node == nullptr, "empty task can't precede any tasks");
  _precede(std::forward<Ts>(tasks)...);
}

// Procedure: reset
inline void TaskBase::reset() {
  _node = nullptr;
}

// Function: empty
inline bool TaskBase::empty() const {
  return _node == nullptr;
}

// Operator
inline TaskBase::operator bool() const {
  return _node != nullptr;
}

// ----------------------------------------------------------------------------

/**
@class PullTask

@brief the handle to a host (cpu) task
*/
class HostTask : public TaskBase {

  friend class FlowBuilder;

  using node_handle_t = Node::Host;
  
  public:

    /**
    @brief constructs an empty host task handle
    */
    HostTask() = default;

    /**
    @brief assigns a work to the host task
    
    @tparam C callable type
    
    @param callable a callable object acceptable to std::function
    */
    template <typename C>
    void work(C&& callable);
    
  private:

    HostTask(Node*);
};

inline HostTask::HostTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

template <typename C>
void HostTask::work(C&& callable) {
  HF_THROW_IF(!_node, "host task is empty");
  _node->_work = std::forward<C>(callable);
}

// ----------------------------------------------------------------------------

/**
@class PullTask

@brief the handle to a pull task
*/
class PullTask : public TaskBase {

  friend class FlowBuilder;
  
  friend PointerCaster to_kernel_argument(PullTask);
  
  using node_handle_t = Node::Pull;

  public:

    /**
    @brief constructs an empty pull task handle
    */
    PullTask() = default;

    /**
    @brief alters the host memory block to copy to gpu

    @tparam T data type of the host memory block
    
    @param source the pointer to the beginning of the host memory block
    @param N number of items of type T to pull

    The number of bytes copied to gpu is equal to sizeof(T)*N.
    It is users' responsibility to ensure the data type and the size are correct.
    */
    template <typename T>
    void pull(const T* source, size_t N);
    
  private:
    
    PullTask(Node*);
    
    void* _d_data();
};

// Constructor
inline PullTask::PullTask(Node* node) : 
  TaskBase::TaskBase {node} {
}
    
// Procedure: pull
template <typename T>
void PullTask::pull(const T* source, size_t N) {

  HF_THROW_IF(!_node, "pull task is empty");

  auto& handle = nonstd::get<node_handle_t>(_node->_handle);
  handle.h_data = source;
  handle.h_size = N * sizeof(T);
}

// Function: d_data
inline void* PullTask::_d_data() {
  return nonstd::get<node_handle_t>(_node->_handle).d_data;
}
  
// ----------------------------------------------------------------------------

/**
@class PushTask

@brief the handle to a push task
*/
class PushTask : public TaskBase {

  friend class FlowBuilder;
  
  using node_handle_t = Node::Push;

  public:

    /**
    @brief constructs an empty push task handle
    */
    PushTask() = default;
    
    /**
    @brief alters the host memory block to push from gpu
    
    @tparam T data type of the host memory block

    @param target the pointer to the beginning of the host memory block
    @param source the source pull task that stores the gpu memory block
    @param N number of items of type T to push

    The number of bytes to copy to the host is sizeof(T)*N. 
    It is users' responsibility to ensure the data type and the size are correct.
    */
    template <typename T>
    void push(T* target, PullTask source, size_t N);

  private:

    PushTask(Node* node);
};

inline PushTask::PushTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

template <typename T>
void PushTask::push(T* target, PullTask source, size_t N) {

  HF_THROW_IF(
    !_node || !source, "both push and pull tasks should be non-empty"
  );
  
  auto& handle = nonstd::get<node_handle_t>(_node->_handle);
  handle.h_data = target,
  handle.source = source._node;
  handle.h_size = N * sizeof(T);
}

// ----------------------------------------------------------------------------

/**
@class KernelTask

@brief the handle to a kernel (gpu) task
*/
class KernelTask : public TaskBase {

  friend class FlowBuilder;
  
  using node_handle_t = Node::Kernel;

  public:

    /**
    @brief constructs an empty kernel task handle
    */
    KernelTask() = default;

    /**
    @brief alters the x dimension of the grid
    */
    void grid_x(size_t x);
    
    /**
    @brief alters the y dimension of the grid
    */
    void grid_y(size_t y);
    
    /**
    @brief alters the z dimension of the grid
    */
    void grid_z(size_t z);
    
    /**
    @brief alters the grid dimension in the form of {x, y, z}
    */
    void grid(const ::dim3& grid);

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
    */
    void block_x(size_t x);
    
    /**
    @brief alters the y dimension of the block
    */
    void block_y(size_t y);
    
    /**
    @brief alters the z dimension of the block
    */
    void block_z(size_t z);
    
    /**
    @brief alters the block dimension in the form of {x, y, z}
    */
    void block(const ::dim3& block);

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


  private:

    KernelTask(Node* node);
};

inline KernelTask::KernelTask(Node* node) : 
  TaskBase::TaskBase {node} {
}

// Procedure: grid_x
inline void KernelTask::grid_x(size_t x) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).grid.x = x;
}

// Procedure: grid_y
inline void KernelTask::grid_y(size_t y) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).grid.y = y;
}

// Procedure: grid_z
inline void KernelTask::grid_z(size_t z) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).grid.z = z;
}

// Procedure: grid
inline void KernelTask::grid(const ::dim3& grid) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).grid = grid;
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
inline void KernelTask::block_x(size_t x) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).block.x = x;
}

// Procedure: block_y
inline void KernelTask::block_y(size_t y) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).block.y = y;
}

// Procedure: block_z
inline void KernelTask::block_z(size_t z) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).block.z = z;
}

// Procedure: block
inline void KernelTask::block(const ::dim3& block) {
  HF_THROW_IF(!_node, "kernel task is empty");
  nonstd::get<node_handle_t>(_node->_handle).block = block;
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

// ----------------------------------------------------------------------------

template <typename T>
auto to_kernel_argument(T&& t) -> decltype(std::forward<T>(t)) { 
  return std::forward<T>(t); 
}

inline PointerCaster to_kernel_argument(PullTask task) { 
  HF_THROW_IF(!task, "pull task is empty");
  return PointerCaster{task._d_data()}; 
}

}  // end of namespace hf -----------------------------------------------------






