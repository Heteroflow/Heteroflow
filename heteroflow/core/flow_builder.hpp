#pragma once

#include "task.hpp"

namespace hf {

/** 
@class FlowBuilder

@brief Building blocks of a task dependency graph.

*/
class FlowBuilder {

  public:

    /**
    @brief creates a placeholder task

    @tparam T task type (PullTask, PushTask, etc.)

    @return task handle of type T
    */
    template <typename T>
    T placeholder();
    
    /**
    @brief creates a host task from a given callable object
    
    @tparam C callable type
    
    @param callable a callable object acceptable to std::function

    @return HostTask handle
    */
    template <typename C>
    HostTask host(C&& callable); 
    
    /**
    @brief creates a pull task that copies a given host memory block to gpu

    @tparam T data type of the host memory block
    
    @param source the pointer to the beginning of the host memory block
    @param N number of items of type T to pull

    @return PullTask handle

    The number of bytes copied to gpu is sizeof(T)*N.
    It is users' responsibility to ensure the data type and the size are correct.
    */
    template <typename T>
    PullTask pull(const T* source, size_t N);
    
    /**
    @brief creates a push task that copies a given gpu memory block to the host
    
    @tparam T data type of the host memory block

    @param target the pointer to the beginning of the host memory block
    @param source the source pull task that stores the gpu memory block
    @param N number of items of type T to push

    @return PushTask handle
    
    The number of bytes copied to the host is sizeof(T)*N. 
    It is users' responsibility to ensure the data type and the size are correct.
    */
    template <typename T>
    PushTask push(T* target, PullTask source, size_t N);
    
    /**
    @brief creates a kernel task that launches a given kernel 

    @tparam ArgsT... argument types

    @param func kernel function
    @param args... arguments to forward to the kernel function

    @return KernelTask handle

    The function performs default configuration to launch the kernel.
    */
    template <typename F, typename... ArgsT>
    KernelTask kernel(F&& func, ArgsT&&... args);

    /**
    @brief clears the graph
    */
    void clear();

  private:

    std::vector<std::unique_ptr<Node>> _nodes;
};

// Function: placeholder
template <typename T>
T FlowBuilder::placeholder() {
  _nodes.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<typename T::node_handle_t>{}
  ));
  return T(_nodes.back().get());
}

// Function: host
template <typename C>
HostTask FlowBuilder::host(C&& callable) {

  _nodes.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Host>{}
  ));
  
  // assign the work
  _nodes.back()->_work = std::forward<C>(callable);

  return HostTask(_nodes.back().get());
}

// Function: pull
template <typename T>
PullTask FlowBuilder::pull(const T* source, size_t N) {
  _nodes.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Pull>{}, source, N
  ));
  return PullTask(_nodes.back().get());
}
    
// Function: push
template <typename T>
PushTask FlowBuilder::push(T* target, PullTask source, size_t N) {

  HF_THROW_IF(!source, "source pull task is empty");

  _nodes.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Push>{}, target, source._node, N
  ));

  return PushTask(_nodes.back().get());
}

// Function: kernel    
template <typename F, typename... ArgsT>
KernelTask FlowBuilder::kernel(F&& func, ArgsT&&... args) {
  
  static_assert(
    function_traits<F>::arity == sizeof...(args), 
    "arguments arity mismatch"
  );

  _nodes.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Kernel>{}
  ));

  auto& node = _nodes.back();
  
  // assign the work
  node->_work = [node=node.get(), func, args...] () {
    auto& h = node->_kernel_handle();
    func<<<h.grid, h.block, h.shm, h.stream>>>(
      to_kernel_argument(args)...
    );
  };

  node->_work();
  
  return KernelTask(node.get());
}

// Procedure: clear
inline void FlowBuilder::clear() {
  _nodes.clear();
}


}  // end of namespace hf -----------------------------------------------------


