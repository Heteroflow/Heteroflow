
// Heteroflow: Parallel CPU-GPU Programming using Task-based Models

namespace hf {

// TaskBase
class TaskBase {

  public:

  virtual ~TaskBase() = default;

  void precede(Task&);

  private:

};

// ----------------------------------------------------------------------------

class HostTask : public HostTask {
  
  public:
    
    template <typename C>
    HostTask(C&& callable);

  private:

};

// ----------------------------------------------------------------------------

// PullTask
class PullTask : public TaskBase {

  public:

  PullTask(void* h_data, size_t h_size);

  ~PullTask();

  void pull(void* h_data, size_t h_size);

  private:
    
  void* _h_data  {nullptr};
  void* _d_data  {nullptr};
  size_t _h_size {0};
  size_t _d_size {0};
};
  
inline PullTask(void* h_data, size_t h_size) : 
  _h_data {h_data}, _h_size {h_size} {
}

inline ~PullTask() {
  // TODO
}

inline void pull(void* h_data, size_t h_size) {
  _h_data = h_data;
  _h_size = h_size;
}

// ----------------------------------------------------------------------------

class PushTask {

  public:

  private:

    PullTask& _source;

    void* _target {nullptr};

    size_t _size {0};
};

// ----------------------------------------------------------------------------

// Class: Heteroflow
class Heteroflow {
  
  public:

  private:
};


}  // end of namespace hf -----------------------------------------------------






