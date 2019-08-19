#pragma once

#include "graph.hpp"

namespace hf {

// Class: TaskBase 
class TaskBase {
  
  friend class HostTask;
  friend class PullTask;
  friend class PushTask;
  friend class KernelTask;
  friend class FlowBuilder;

  public:

    /**
    @brief constructs an empty taskbase object
    */
    TaskBase() = default;

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

// Class: HostTask
class HostTask : public TaskBase {

  friend class FlowBuilder;
  
  public:

    HostTask() = default;
    
  private:

    HostTask(Node*);
};

inline HostTask::HostTask(Node* node) : 
  TaskBase::TaskBase {node} {
}


// ----------------------------------------------------------------------------

// PullTask
class PullTask : public TaskBase {

  friend class FlowBuilder;

  public:

    PullTask() = default;

  private:
    
    PullTask(Node*);
};

inline PullTask::PullTask(Node* node) : 
  TaskBase::TaskBase {node} {
}
  
// ----------------------------------------------------------------------------

// Class: PushTask
class PushTask : public TaskBase {

  friend class FlowBuilder;

  public:

    PushTask() = default;

  private:

    PushTask(Node* node);
};

inline PushTask::PushTask(Node* node) : 
  TaskBase::TaskBase {node} {
}


// ----------------------------------------------------------------------------


}  // end of namespace hf -----------------------------------------------------






