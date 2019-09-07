#pragma once

#include <thread>
#include <mutex>
#include <random>

#include "../facility/notifier.hpp"
#include "../facility/spmc_queue.hpp"

#include "topology.hpp"
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
  
  struct Device {
    Device(int);
    ~Device();
    int id;
    cudaStream_t push_stream;
    cudaStream_t pull_stream;
    cudaStream_t kernel_stream;
  };

  public:
    
    /**
    @brief constructs the executor with N workers and M GPUs
    */
    explicit Executor(
      unsigned N = std::thread::hardware_concurrency(), 
      unsigned M = cuda::num_devices()
    );
    
    /**
    @brief destructs the executor 
    */
    ~Executor();
    
    /**
    @brief runs the heteroflow once
    
    @param heteroflow a tf::Heteroflow object

    @return a std::future to access the execution status of the heteroflow
    */
    std::future<void> run(Heteroflow& heteroflow);

    /**
    @brief runs the heteroflow once and invoke a callback upon completion

    @param heteroflow a tf::Heteroflow object 
    @param callable a callable object to be invoked after this run

    @return a std::future to access the execution status of the heteroflow
    */
    template<typename C>
    std::future<void> run(Heteroflow& heteroflow, C&& callable);

    /**
    @brief runs the heteroflow for N times
    
    @param heteroflow a tf::Heteroflow object
    @param N number of runs

    @return a std::future to access the execution status of the heteroflow
    */
    std::future<void> run_n(Heteroflow& heteroflow, size_t N);

    /**
    @brief runs the heteroflow for N times and then invokes a callback

    @param heteroflow a tf::Heteroflow 
    @param N number of runs
    @param callable a callable object to be invoked after this run

    @return a std::future to access the execution status of the heteroflow
    */
    template<typename C>
    std::future<void> run_n(Heteroflow& heteroflow, size_t N, C&& callable);

    /**
    @brief runs the heteroflow multiple times until the predicate becomes true and 
           then invokes a callback

    @param heteroflow a tf::Heteroflow 
    @param pred a boolean predicate to return true for stop

    @return a std::future to access the execution status of the heteroflow
    */
    template<typename P>
    std::future<void> run_until(Heteroflow& heteroflow, P&& pred);

    /**
    @brief runs the heteroflow multiple times until the predicate becomes true and 
           then invokes the callback

    @param heteroflow a tf::Heteroflow 
    @param pred a boolean predicate to return true for stop
    @param callable a callable object to be invoked after this run

    @return a std::future to access the execution status of the heteroflow
    */
    template<typename P, typename C>
    std::future<void> run_until(Heteroflow& heteroflow, P&& pred, C&& callable);

    /**
    @brief queries the number of workers
    */
    size_t num_workers() const;
    
    /**
    @brief queries the number of gpu devices
    */
    size_t num_devices() const;
    
    /**
    @brief wait for all pending graphs to complete
    */
    void wait_for_all();

  private:
    
    std::condition_variable _topology_cv;
    std::mutex _topology_mutex;
    std::mutex _queue_mutex;

    unsigned _num_topologies {0};
    
    // scheduler field
    std::vector<Worker> _workers;
    std::vector<Notifier::Waiter> _waiters;
    std::vector<std::thread> _threads;
    std::vector<Device> _devices;

    WorkStealingQueue<Node*> _queue;

    std::atomic<size_t> _num_actives {0};
    std::atomic<size_t> _num_thieves {0};
    std::atomic<bool>   _done        {0};

    Notifier _notifier;
    
    PerThread& _per_thread() const;

    bool _wait_for_task(unsigned me, nonstd::optional<Node*>&);

    void _spawn();
    void _exploit_task(unsigned, nonstd::optional<Node*>&);
    void _explore_task(unsigned, nonstd::optional<Node*>&);
    void _schedule(Node*, bool);
    void _schedule(std::vector<Node*>&);
    void _invoke(unsigned, Node*);
    void _increment_topology();
    void _decrement_topology();
    void _decrement_topology_and_notify();
    void _tear_down_topology(Topology*); 
};

// ----------------------------------------------------------------------------
// Executor::Device
// ----------------------------------------------------------------------------

// Constructor
inline Executor::Device::Device(int dev) :
  id {dev} {
  
  HF_WITH_CUDA_DEVICE(id) {
    HF_CHECK_CUDA(::cudaStreamCreate(&pull_stream), 
      "failed to create pull stream for device ", id
    );
    HF_CHECK_CUDA(::cudaStreamCreate(&push_stream), 
      "failed to create push stream for device ", id
    );
    HF_CHECK_CUDA(::cudaStreamCreate(&kernel_stream), 
      "failed to create kernel stream for device ", id
    );
  }
}

// Destructor
inline Executor::Device::~Device() {
  HF_WITH_CUDA_DEVICE(id) {
    HF_CHECK_CUDA(::cudaStreamSynchronize(pull_stream),
      "failed to sync pull stream ", pull_stream, " on device ", id
    );
    HF_CHECK_CUDA(::cudaStreamDestroy(pull_stream),
      "failed to destroy pull stream ", pull_stream, " on device ", id
    );
    HF_CHECK_CUDA(::cudaStreamSynchronize(push_stream),
      "failed to sync push stream ", push_stream, " on device ", id
    );
    HF_CHECK_CUDA(::cudaStreamDestroy(push_stream),
      "failed to destroy push stream ", push_stream, " on device ", id
    );
    HF_CHECK_CUDA(::cudaStreamSynchronize(kernel_stream),
      "failed to sync kernel stream ", kernel_stream, " on device ", id
    );
    HF_CHECK_CUDA(::cudaStreamDestroy(kernel_stream),
      "failed to destroy kernel stream ", kernel_stream, " on device ", id
    );
  }
}

// ----------------------------------------------------------------------------
// Executor
// ----------------------------------------------------------------------------

// Constructor
inline Executor::Executor(unsigned N, unsigned M) : 
  _workers  {N},
  _waiters  {N},
  _notifier {_waiters} {

  // invalid number
  auto num_devices = cuda::num_devices();
  HF_THROW_IF(num_devices < M, "max device count is ", num_devices);

  // set up the devices
  _devices.reserve(M);
  for(int i=0; i<M; ++i) {
    _devices.emplace_back(i);
  }

  // set up the workers
  _spawn();
}

// Destructor
inline Executor::~Executor() {
  
  // wait for all topologies to complete
  wait_for_all();
  
  // shut down the scheduler
  _done = true;
  _notifier.notify(true);
  
  for(auto& t : _threads){
    t.join();
  } 
}

// Function: num_workers
inline size_t Executor::num_workers() const {
  return _workers.size();
}

// Function: num_devices
inline size_t Executor::num_devices() const {
  return _devices.size();
}

// Function: _per_thread
inline Executor::PerThread& Executor::_per_thread() const {
  thread_local PerThread pt;
  return pt;
}

// Procedure: _spawn
inline void Executor::_spawn() {
  
  for(unsigned i=0; i<_workers.size(); ++i) {
    _threads.emplace_back([this, i] () -> void {

      PerThread& pt = _per_thread();  
      pt.pool = this;
      pt.worker_id = i;
    
      nonstd::optional<Node*> t;
      
      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        _exploit_task(i, t);

        // wait for tasks
        if(_wait_for_task(i, t) == false) {
          break;
        }
      }
      
    });     
  }
}

// Procedure: _exploit_task
inline void Executor::_exploit_task(unsigned i, nonstd::optional<Node*>& t) {
  
  assert(!_workers[i].cache);

  if(t) {
    auto& worker = _workers[i];
    if(_num_actives.fetch_add(1) == 0 && _num_thieves == 0) {
      _notifier.notify(false);
    }
    do {

      _invoke(i, *t);

      if(worker.cache) {
        t = worker.cache;
        worker.cache = nullptr;
      }
      else {
        t = worker.queue.pop();
      }

    } while(t);

    --_num_actives;
  }
}

// Function: _explore_task
inline void Executor::_explore_task(unsigned thief, nonstd::optional<Node*>& t) {
  
  assert(!t);

  const unsigned l = 0;
  const unsigned r = _workers.size() - 1;

  const size_t F = (_workers.size() + 1) << 1;
  const size_t Y = 100;

  size_t f = 0;
  size_t y = 0;

  // explore
  while(!_done) {
  
    unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
      _workers[thief].rdgen
    );
      
    t = (vtm == thief) ? _queue.steal() : _workers[vtm].queue.steal();

    if(t) {
      break;
    }

    if(f++ > F) {
      std::this_thread::yield();
      if(y++ > Y) {
        break;
      }
    }
  }

}

// Function: _wait_for_task
inline bool Executor::_wait_for_task(unsigned me, nonstd::optional<Node*>& t) {

  wait_for_task:

  assert(!t);

  ++_num_thieves;

  explore_task:

  _explore_task(me, t);

  if(t) {
    if(_num_thieves.fetch_sub(1) == 1) {
      _notifier.notify(false);
    }
    return true;
  }

  _notifier.prepare_wait(&_waiters[me]);
  
  if(!_queue.empty()) {

    _notifier.cancel_wait(&_waiters[me]);
    
    t = _queue.steal();
    if(t) {
      if(_num_thieves.fetch_sub(1) == 1) {
        _notifier.notify(false);
      }
      return true;
    }
    else {
      goto explore_task;
    }
  }

  if(_done) {
    _notifier.cancel_wait(&_waiters[me]);
    _notifier.notify(true);
    --_num_thieves;
    return false;
  }

  if(_num_thieves.fetch_sub(1) == 1 && _num_actives) {
    _notifier.cancel_wait(&_waiters[me]);
    goto wait_for_task;
  }
    
  // Now I really need to relinguish my self to others
  _notifier.commit_wait(&_waiters[me]);

  return true;
}

// Procedure: _invoke
inline void Executor::_invoke(unsigned me, Node* node) {
  // TODO
  
  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology clear.
  const auto num_successors = node->num_successors();
  
  // device assignment for gpu task only
  if(node->is_kernel() || node->is_pull()) {
    
    auto root = node->_root();
    auto& dev = root->_device();
    
    // Find a device for this node
    int t = dev;

    if(t == -1) {

      int s = -1;
      int l = 0;
      int r = _devices.size();

      t = std::uniform_int_distribution<int>{l, r-1}(_workers[me].rdgen);

      if(!dev.compare_exchange_strong(s, t)) {
        t = s;
      }
    }

    // kernel task
    if(node->is_kernel()) {
      node->_kernel_handle().device = t;
    }

    // pull task
    if(node->is_pull()) {
      node->_pull_handle().device = t;
    }
    
    printf("task %s is mapped to device %d\n", node->_name.c_str(), t);
  }
  
  // Invoke the work at the node 
  node->_work();
  
  // At this point, the node storage might be destructed.
  Node* cache {nullptr};

  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_num_dependents) == 0) {
      if(cache) {
        _schedule(cache, false);
      }
      cache = node->_successors[i];
    }
  }

  if(cache) {
    _schedule(cache, true);
  }

  // A node without any successor should check the termination of topology
  if(num_successors == 0) {
    if(--(node->_topology->_num_sinks) == 0) {
      _tear_down_topology(node->_topology);
    }
  }
}

// Procedure: _schedule
// The main procedure to schedule a give task node.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(Node* node, bool bypass) {
  
  assert(_workers.size() != 0);
  
  // caller is a worker to this pool
  auto& pt = _per_thread();

  if(pt.pool == this) {
    if(!bypass) {
      _workers[pt.worker_id].queue.push(node);
    }
    else {
      assert(!_workers[pt.worker_id].cache);
      _workers[pt.worker_id].cache = node;
    }
    return;
  }

  // other threads
  {
    std::lock_guard<std::mutex> lock(_queue_mutex);
    _queue.push(node);
  }

  _notifier.notify(false);
}

// Procedure: _schedule
// The main procedure to schedule a set of task nodes.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(std::vector<Node*>& nodes) {

  assert(_workers.size() != 0);
  
  // We need to cacth the node count to avoid accessing the nodes
  // vector while the parent topology is removed!
  const auto num_nodes = nodes.size();
  
  if(num_nodes == 0) {
    return;
  }

  // worker thread
  auto& pt = _per_thread();

  if(pt.pool == this) {
    for(size_t i=0; i<num_nodes; ++i) {
      _workers[pt.worker_id].queue.push(nodes[i]);
    }
    return;
  }
  
  // other threads
  {
    std::lock_guard<std::mutex> lock(_queue_mutex);
    for(size_t k=0; k<num_nodes; ++k) {
      _queue.push(nodes[k]);
    }
  }

  if(num_nodes >= _workers.size()) {
    _notifier.notify(true);
  }
  else {
    for(size_t k=0; k<num_nodes; ++k) {
      _notifier.notify(false);
    }
  }
}

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Topology* tpg) {

  auto &f = tpg->_heteroflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(!tpg->_pred()) {
    tpg->_recover_num_sinks();
    _schedule(tpg->_sources); 
  }
  // case 2: the final run of this topology
  else {
    
    if(tpg->_call != nullptr) {
      tpg->_call();
    }

    f._mtx.lock();

    // If there is another run (interleave between lock)
    if(f._topologies.size() > 1) {

      // Set the promise
      tpg->_promise.set_value();
      f._topologies.pop_front();
      f._mtx.unlock();
      
      // decrement the topology but since this is not the last we don't notify
      _decrement_topology();

      f._topologies.front()._bind(f._graph);
      _schedule(f._topologies.front()._sources);
    }
    else {
      assert(f._topologies.size() == 1);

      // Need to back up the promise first here becuz heteroflow might be 
      // destroy before heteroflow leaves
      auto p {std::move(tpg->_promise)};

      f._topologies.pop_front();

      f._mtx.unlock();

      // We set the promise in the end in case heteroflow leaves before heteroflow
      p.set_value();

      _decrement_topology_and_notify();
    }
  }
}

// Procedure: _increment_topology
inline void Executor::_increment_topology() {
  std::lock_guard<std::mutex> lock(_topology_mutex);
  ++_num_topologies;
}

// Procedure: _decrement_topology_and_notify
inline void Executor::_decrement_topology_and_notify() {
  std::lock_guard<std::mutex> lock(_topology_mutex);
  if(--_num_topologies == 0) {
    _topology_cv.notify_all();
  }
}

// Procedure: _decrement_topology
inline void Executor::_decrement_topology() {
  std::lock_guard<std::mutex> lock(_topology_mutex);
  --_num_topologies;
}

// Procedure: wait_for_all
inline void Executor::wait_for_all() {
  std::unique_lock<std::mutex> lock(_topology_mutex);
  _topology_cv.wait(lock, [&](){ return _num_topologies == 0; });
}

// Function: run
inline std::future<void> Executor::run(Heteroflow& f) {
  return run_n(f, 1, [](){});
}

// Function: run
template <typename C>
std::future<void> Executor::run(Heteroflow& f, C&& c) {
  return run_n(f, 1, std::forward<C>(c));
}

// Function: run_n
inline std::future<void> Executor::run_n(Heteroflow& f, size_t repeat) {
  return run_n(f, repeat, [](){});
}

// Function: run_n
template <typename C>
std::future<void> Executor::run_n(Heteroflow& f, size_t repeat, C&& c) {
  return run_until(f, [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c));
}

// Function: run_until    
template<typename P>
std::future<void> Executor::run_until(Heteroflow& f, P&& pred) {
  return run_until(f, std::forward<P>(pred), [](){});
}

// Function: run_until
template<typename P, typename C>
std::future<void> Executor::run_until(Heteroflow& f, P&& pred, C&& c) {

  assert(_workers.size() > 0);
  
  _increment_topology();
  
  // device mapping
  for(auto& node : f._graph) {
    if(node->is_kernel()) {
      auto& h = node->_kernel_handle();
      for(auto s : h.sources) {
        node->_union(s);
      }
    }
  }
  
  for(auto& node : f._graph) {
    if(node->is_kernel() || node->is_pull()) {
      std::cout << node->_name << ' ' << node->_root()->_name << std::endl;
    }
  }

  // Special case of predicate
  if(pred()) {
    std::promise<void> promise;
    promise.set_value();
    _decrement_topology_and_notify();
    return promise.get_future();
  }
  
  /*// Special case of zero workers requires:
  //  - iterative execution to avoid stack overflow
  //  - avoid execution of last_work
  if(_workers.size() == 0 || f.empty()) {
    
    Topology tpg(f, std::forward<P>(pred), std::forward<C>(c));

    // Clear last execution data & Build precedence between nodes and target
    tpg._bind(f._graph);

    std::stack<Node*> stack;

    do {
      _schedule_unsync(tpg._sources, stack);
      while(!stack.empty()) {
        auto node = stack.top();
        stack.pop();
        _invoke_unsync(node, stack);
      }
      tpg._recover_num_sinks();
    } while(!std::invoke(tpg._pred));

    if(tpg._call != nullptr) {
      std::invoke(tpg._call);
    }

    tpg._promise.set_value();
    
    _decrement_topology_and_notify();
    
    return tpg._promise.get_future();
  }*/
  
  // Multi-threaded execution.
  bool run_now {false};
  Topology* tpg;
  std::future<void> future;
  
  {
    std::lock_guard<std::mutex> lock(f._mtx);

    // create a topology for this run
    f._topologies.emplace_back(f, std::forward<P>(pred), std::forward<C>(c));
    tpg = &(f._topologies.back());
    future = tpg->_promise.get_future();
    
    if(f._topologies.size() == 1) {
      run_now = true;
      //tpg->_bind(f._graph);
      //_schedule(tpg->_sources);
    }
  }
  
  // Notice here calling schedule may cause the topology to be removed sonner 
  // before the function leaves.
  if(run_now) {
    tpg->_bind(f._graph);
    _schedule(tpg->_sources);
  }

  return future;
}


}  // end of namespace hf -----------------------------------------------------




