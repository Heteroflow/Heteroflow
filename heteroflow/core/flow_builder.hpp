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

    @tparam T task type (SpanTask, CopyTask, KernelTask, and HostTask)

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
    @brief creates a span task that copies a given host memory block to gpu

    @tparam ArgsT argements types
    @param args arguments to forward to construct a span task

    @return SpanTask handle
    */
    template <typename... ArgsT>
    SpanTask span(ArgsT&&... args);
    
    /**
    @brief creates a copy task that copies a given gpu memory block to the host
    
    @tparam ArgsT argements types
    @param args arguments to forward to construct a copy task

    @return CopyTask handle
    */
    template <typename... ArgsT>
    CopyTask copy(ArgsT&&... args);
    
    /**
    @brief creates a kernel task that launches a given kernel 
    
    @tparam ArgsT argements types
    @param args arguments to forward to construct a kernel task

    @return KernelTask handle
    */
    template <typename... ArgsT>
    KernelTask kernel(ArgsT&&... args);

    /**
    @brief creates a fill task that send data between span tasks
    
    @tparam ArgsT argements types
    @param args arguments to forward to construct a fill task
    
    @return FillTask handle
    */
    template <typename... ArgsT>
    FillTask fill(ArgsT&&... args);

    /**
    @brief clears the graph
    */
    void clear();
    
    /**
    @brief queries the number of nodes
    */
    size_t size() const;
    
    /**
    @brief queries the number of nodes
    */
    bool empty() const;
    
    /**
    @brief constructs a task dependency graph of range-based parallel_for
    
    The task dependency graph applies a callable object 
    to the dereferencing of every iterator 
    in the range [beg, end) one by one.

    @tparam I input iterator type
    @tparam C callable type

    @param beg iterator to the beginning (inclusive)
    @param end iterator to the end (exclusive)
    @param c a callable object to be applied to 

    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename C>
    std::pair<HostTask, HostTask> parallel_for(I beg, I end, C&& c);
    
    /**
    @brief constructs a task dependency graph of range-based parallel_for
    
    The task dependency graph applies a callable object 
    to each indexed item in the range [beg, end) one by one.

    @tparam I input iterator type
    @tparam C callable type

    @param beg iterator to the beginning (inclusive)
    @param end iterator to the end (exclusive)
    @param step step between successive iterations
    @param c a callable object to be applied to 

    @return a pair of Task handles to the beginning and the end of the graph
    */
    template <typename I, typename C>
    std::pair<HostTask, HostTask> parallel_for(I beg, I end, I step, C&& c);

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

// Function: span
template <typename... C>
SpanTask FlowBuilder::span(C&&... c) {
  
  _graph.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Span>{}
  ));

  return SpanTask(_graph.back().get())
        .span(std::forward<C>(c)...);
}

// Function: copy
template <typename... ArgsT>
CopyTask FlowBuilder::copy(ArgsT&&... args) {

  _graph.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Copy>{}
  ));

  return CopyTask(_graph.back().get())
        .copy(std::forward<ArgsT>(args)...);
}


// Function: fill 
template <typename... ArgsT>
FillTask FlowBuilder::fill(ArgsT&&... args) {

  _graph.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Fill>{}
  ));

  return FillTask(_graph.back().get())
        .fill(std::forward<ArgsT>(args)...);
}


// Function: kernel    
template <typename... ArgsT>
KernelTask FlowBuilder::kernel(ArgsT&&... args) {
  
  _graph.emplace_back(std::make_unique<Node>(
    nonstd::in_place_type_t<Node::Kernel>{}
  ));

  return KernelTask(_graph.back().get())
        .kernel(std::forward<ArgsT>(args)...);
}

// Procedure: clear
inline void FlowBuilder::clear() {
  _graph.clear();
}

// Function: size
inline size_t FlowBuilder::size() const {
  return _graph.size();
}

// Function: empty
inline bool FlowBuilder::empty() const {
  return _graph.empty();
}

// Function: parallel_for
template <typename I, typename C>
std::pair<HostTask, HostTask> FlowBuilder::parallel_for(I beg, I end, C&& c) {
  
  auto S = placeholder<HostTask>();
  auto T = placeholder<HostTask>();
  
  while(beg != end) {
    auto task = host([itr=beg, c] () mutable {
      c(*itr);
    });
    S.precede(task);
    ++beg;
  }

  if(S.num_successors() == 0) {
    S.precede(T);
  }
  
  return std::make_pair(S, T); 

}

// Function: parallel_for
template <typename I, typename C>
std::pair<HostTask, HostTask> FlowBuilder::parallel_for(I beg, I end, I s, C&& c){

  auto S = placeholder<HostTask>();
  auto T = placeholder<HostTask>();

  for(auto i=beg; i<end; i+=s) {
    auto task = host([i, c] () mutable {
      c(i);
    });
    S.precede(task);
    task.precede(T);
  }

  if(S.num_successors() == 0) {
    S.precede(T);
  }
  
  return std::make_pair(S, T); 
}


}  // end of namespace hf -----------------------------------------------------


