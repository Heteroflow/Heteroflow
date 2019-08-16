
// Heteroflow: Parallel CPU-GPU Programming using Task-based Models

namespace hf {

// Task
class Task {

  struct Host {
  };

  struct Pull {
    void* d_data {nullptr};
    void* h_data {nullptr};
    size_t d_size {0};
    size_t h_size {0};
  };

  struct Push {
  };

  public:

    void precede(Task&);

  private:

};

// ----------------------------------------------------------------------------

// Class: Heteroflow
class Heteroflow {
  
  public:

  private:
};


}  // end of namespace hf -----------------------------------------------------






