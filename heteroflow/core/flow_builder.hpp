#pragma once

#include "task.hpp"

namespace hf {

using Graph = std::vector<std::unique_ptr<Node>>;

/** 
@class FlowBuilder

@brief Building blocks of a task dependency graph.

*/
class FlowBuilder {

  friend class Heteroflow;
  friend class Executor;

  public:

    /**
    @brief creates a placeholder task

    @tparam T task type (PullTask, PushTask, KernelTask, and HostTask)

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

    @param source the pointer to the beginning of the host memory block
    @param N number of bytes to pull

    @return PullTask handle
    */
    PullTask pull(const void* source, size_t N);
    
    /**
    @brief creates a push task that copies a given gpu memory block to the host
    
    @param target the pointer to the beginning of the host memory block
    @param source the source pull task that stores the gpu memory block
    @param N number of bytes to push

    @return PushTask handle
    */
    PushTask push(void* target, PullTask source, size_t N);
    
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
    
    /**
    @brief queries the number of nodes
    */
    size_t num_nodes() const;

  private:

    Graph _graph;
};

// Function: placeholder
template <typename T>
T FlowBuilder::placeholder() {
  _graph.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<typename T::node_handle_t>{}
  ));
  return T(_graph.back().get());
}

// Function: host
template <typename C>
HostTask FlowBuilder::host(C&& callable) {

  _graph.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Host>{}
  ));
  
  // assign the work
  return HostTask(_graph.back().get())
        .work(std::forward<C>(callable));
}

// Function: pull
inline PullTask FlowBuilder::pull(const void* source, size_t N) {
  
  _graph.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Pull>{}, source, N
  ));

  PullTask task(_graph.back().get());

  task._make_work();

  return task;
}

// Function: push
inline PushTask FlowBuilder::push(void* target, PullTask source, size_t N) {

  HF_THROW_IF(!source, "source pull task is empty");

  _graph.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Push>{}, target, source._node, N
  ));

  PushTask task(_graph.back().get());

  task._make_work();

  return task;
}

// Function: kernel    
template <typename F, typename... ArgsT>
KernelTask FlowBuilder::kernel(F&& func, ArgsT&&... args) {
  
  static_assert(
    function_traits<F>::arity == sizeof...(args), 
    "argument arity mismatches"
  );

  _graph.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Kernel>{}
  ));

  KernelTask task(_graph.back().get());

  task.kernel(std::forward<F>(func), std::forward<ArgsT>(args)...);

  return task;
}

// Procedure: clear
inline void FlowBuilder::clear() {
  _graph.clear();
}

// Function: num_nodes
inline size_t FlowBuilder::num_nodes() const {
  return _graph.size();
}

}  // end of namespace hf -----------------------------------------------------


