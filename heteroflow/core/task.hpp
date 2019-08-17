#pragma once

#include <iostream>
#include <string>
#include <atomic>
#include <vector>
#include <functional>

#include "../facility/error.hpp"

namespace hf {

/**
@class TaskBase

@brief Base class from which all tasks are derived

TaskBase manages common data members required for all tasks.

*/
class TaskBase {

  friend class HostTask;
  friend class KernelTask;
  friend class PullTask;
  friend class PushTask;

  public:
    
    /**
    @brief virtual destructor
    */
    virtual ~TaskBase() = default;

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
    

  private:

    std::string _name;

    std::vector<TaskBase*> _successors;
    std::vector<TaskBase*> _dependents;
    
    std::atomic<int> _num_dependents {0};
    
    template <typename T>
    void _precede(T&&);
    
    template <typename T, typename... Rest>
    void _precede(T&&, Rest&&...);
};

// Procedure: name
template <typename S>
void TaskBase::name(S&& name) {
  _name = std::forward<S>(name);
}

// Function: name
inline const std::string& TaskBase::name() const {
  return _name;
}

// Function: num_successors
inline size_t TaskBase::num_successors() const {
  return _successors.size();
}

// Function: num_dependents
inline size_t TaskBase::num_dependents() const {
  return _dependents.size();
}

// Procedure: precede
template <typename T>
void TaskBase::_precede(T&& other) {
  _successors.push_back(other);
  other->_dependents.push_back(this);
  other->_num_dependents.fetch_add(1, std::memory_order_relaxed);
}

// Procedure: _precede
template <typename T, typename... Ts>
void TaskBase::_precede(T&& task, Ts&&... others) {
  _precede(std::forward<T>(task));
  _precede(std::forward<Ts>(others)...);
}

// Function: precede
template <typename... Ts>
void TaskBase::precede(Ts&&... tgts) {
  //(_precede(tgts), ...);  C++17 fold expression
  _precede(std::forward<Ts>(tgts)...);
}

// ----------------------------------------------------------------------------

// Class: HostTask
class HostTask : public TaskBase {
  
  public:
    
    template <typename C>
    HostTask(C&& callable);

  private:

};

// ----------------------------------------------------------------------------

// PullTask
class PullTask : public TaskBase {

  public:
    
    template <typename T>
    PullTask(const T* h_data, size_t h_size);

    ~PullTask();
      
  private:
    
    const void* _h_data {nullptr};
    void*       _d_data {nullptr};

    size_t _h_bytes {0};
    size_t _d_bytes {0};
};
  
template <typename T>
PullTask::PullTask(const T* h_data, size_t h_size) : 
  _h_data {h_data}, _h_bytes {h_size*sizeof(T)} {
}

inline PullTask::~PullTask() {
  // TODO
}

// ----------------------------------------------------------------------------

// Class: PushTask
class PushTask : public TaskBase {

  public:

    PushTask(void* target, PullTask& source, size_t size);

  private:

    void* _target {nullptr};
    
    PullTask& _source;

    size_t _size {0};
};

// Constructor
inline PushTask::PushTask(void* target, PullTask& source, size_t size) : 
  _target {target},
  _source {source},
  _size   {size}
{
}

// ----------------------------------------------------------------------------

// Class: Heteroflow
class Heteroflow {
  
  public:

  private:

    std::string _name;


};


}  // end of namespace hf -----------------------------------------------------






