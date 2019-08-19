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
    
    @param h_data the pointer to the beginning of the host memory block
    @param h_size number of items of type T to pull

    @return PullTask handle

    The number of bytes copied to gpu is sizeof(T)*h_size.
    It is users' responsibility to ensure the data type and size are correct.
    */
    template <typename T>
    PullTask pull(const T* h_data, size_t h_size);
    
    /**
    @brief creates a push task that copies a given gpu memory block to the host
    
    @tparam T data type of the host memory block

    @param h_data the pointer to the beginning of the host memory block
    @param source the source pull task that stores the gpu memory block
    @param h_size number of items of type T to push

    @return PushTask handle
    
    The number of bytes copied to the host is sizeof(T)*h_size. 
    It is users' responsibility to ensure the data type and size are correct.
    */
    template <typename T>
    PushTask push(T* h_data, PullTask source, size_t h_size);

    /**
    @brief clears the graph
    */
    void clear();

  private:

    std::vector<std::unique_ptr<Node>> _nodes;
};

// Function: host
template <typename C>
HostTask FlowBuilder::host(C&& callable) {
  _nodes.emplace_back(std::make_unique<Node>(
    mpark::in_place_type_t<Node::Host>{}, std::forward<C>(callable)
  ));
  return HostTask(_nodes.back().get());
}

// Function: pull
template <typename T>
PullTask FlowBuilder::pull(const T* h_data, size_t h_size) {
  _nodes.emplace_back(std::make_unique<Node>(
    mpark::in_place_type_t<Node::Pull>{}, h_data, h_size
  ));
  return PullTask(_nodes.back().get());
}
    
// Function: push
template <typename T>
PushTask FlowBuilder::push(T* h_data, PullTask source, size_t h_size) {

  HF_THROW_IF(!source, "source pull task is an empty task");

  _nodes.emplace_back(std::make_unique<Node>(
    mpark::in_place_type_t<Node::Push>{}, h_data, source._node, h_size
  ));

  return PushTask(_nodes.back().get());
}

// Procedure: clear
inline void FlowBuilder::clear() {
  _nodes.clear();
}


}  // end of namespace hf -----------------------------------------------------


