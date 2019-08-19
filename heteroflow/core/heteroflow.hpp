#pragma once

#include "flow_builder.hpp"

namespace hf {

/**
@class Heteroflow

@brief the class to create a task dependency graph

*/
class Heteroflow : public FlowBuilder {

  friend class Executor;

  public:
    
    /**
    @brief constructs a heteroflow
    */
    Heteroflow() = default;
    
    /**
    @brief destroy the heteroflow (virtual call)
    */
    virtual ~Heteroflow();
    
    /**
    @brief assigns a name to the heteroflow
    
    @tparam S string type

    @param name a @std_string acceptable string
    */
    template <typename S>
    void name(S&& name);

    /**
    @brief queries the name of the heteroflow
    */
    const std::string& name() const ;

  private:

    std::string _name;

};

// Destructor
inline Heteroflow::~Heteroflow() {
}

// Procedure: name
template <typename S>
void Heteroflow::name(S&& name) {
  _name = std::forward<S>(name);
}

// Function: name
inline const std::string& Heteroflow::name() const {
  return _name;
}

}  // end of namespace hf -----------------------------------------------------
