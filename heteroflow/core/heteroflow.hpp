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
    @brief constructs a heteroflow with a given name

    @param name a @std_string acceptable string
    */
    template <typename S>
    Heteroflow(S&& name);
    
    /**
    @brief destroy the heteroflow (virtual call)
    */
    virtual ~Heteroflow();
    
    /**
    @brief dumps the heteroflow to a std::ostream in DOT format

    @param ostream a std::ostream target
    */
    void dump(std::ostream& ostream) const;
    
    /**
    @brief dumps the heteroflow in DOT format to a std::string
    */
    std::string dump() const;
    
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
    const std::string& name() const;

  private:

    std::string _name;
    
    std::mutex _mtx;

    std::list<Topology> _topologies;
};

// Constructor
template <typename S>
Heteroflow::Heteroflow(S&& name) : _name {std::forward<S>(name)} {
}

// Destructor
inline Heteroflow::~Heteroflow() {
  assert(_topologies.empty());

  for(auto& n : _graph) {
    auto idx = n->_handle.index();
    if(idx == Node::PULL_IDX) {
      auto& h = n->_pull_handle();
      assert(h.device == -1);
      assert(h.d_data == nullptr);
      assert(h.d_size == 0);
      //assert(h.height == 0);
      //assert(h.parent == nullptr);
    }
    else if(idx == Node::PUSH_IDX) {
    }
    else if(idx == Node::KERNEL_IDX) {
      auto& h = n->_kernel_handle();
      assert(h.device == -1);
      //assert(h.height == 0);
      //assert(h.parent == nullptr);
    }
    assert(n->_height == 0);
    assert(n->_parent == n.get());
  }
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

// Function: dump
inline std::string Heteroflow::dump() const {
  std::ostringstream oss;
  dump(oss);
  return oss.str();
}

// Procedure: dump
inline void Heteroflow::dump(std::ostream& os) const {

  os << "digraph ";
  
  if(_name.empty()) {
    os << 'p' << this;
  }
  else {
    os << _name;
  }

  os << " {\nrankdir=\"TB\";\n";
  
  // dump the details of this taskflow
  for(const auto& n : _graph) {
    // regular task
    n->dump(os);
  }
  
  os << "}\n";
}


}  // end of namespace hf -----------------------------------------------------



