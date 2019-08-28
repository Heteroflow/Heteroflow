#pragma once

#include <random>

#include "../facility/notifier.hpp"
#include "../facility/spmc_queue.hpp"
#include "heteroflow.hpp"

namespace hf {

/** @class Executor

@brief The executor class to run a heteroflow graph.

An executor object manages a set of worker threads and implements 
an efficient CPU-GPU co-scheduling algorithm to execute task graphs.

*/
class Executor {
  
  struct Worker {
    std::mt19937 rdgen { std::random_device{}() };
    WorkStealingQueue<Node*> queue;
    Node* cache {nullptr};
  };
    
  struct PerThread {
    Executor* pool {nullptr}; 
    int worker_id  {-1};
  };

  public:
    
    /**
    @brief constructs the executor with N workers and M GPUs
    */
    explicit Executor(unsigned N, unsigned M);
    
    /**
    @brief destructs the executor 
    */
    ~Executor();

  private:
    
    std::condition_variable _topology_cv;
    std::mutex _topology_mutex;
    std::mutex _queue_mutex;

    unsigned _num_topologies {0};
    
    // scheduler field
    std::vector<Worker> _workers;
    std::vector<Notifier::Waiter> _waiters;
    std::vector<std::thread> _threads;

    WorkStealingQueue<Node*> _queue;

    std::atomic<size_t> _num_actives {0};
    std::atomic<size_t> _num_thieves {0};
    std::atomic<bool>   _done        {0};

    Notifier _notifier;

};

// Constructor
inline Executor::Executor(unsigned N, unsigned M) : 
  _workers  {N},
  _waiters  {N},
  _notifier {_waiters} {
  //_spawn(N);
}

}  // end of namespace hf -----------------------------------------------------




