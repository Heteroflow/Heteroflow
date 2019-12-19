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
    unsigned id;
    Domain domain;
    std::mt19937 rdgen { std::random_device{}() };
    WorkStealingQueue<Node*> cpu_queue;
    WorkStealingQueue<Node*> gpu_queue;
    Node* cache {nullptr};
  };
    
  struct PerThread {
    Worker* worker {nullptr};
  };
  
  struct Device {
    int id {-1};
    cuda::Allocator allocator;
    std::atomic<int> load {0};
    std::vector<cudaStream_t> cstreams;
    std::vector<cudaStream_t> kstreams;
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
    @brief queries the number of cpu workers
    */
    size_t num_cpu_workers() const;
    
    /**
    @brief queries the number of gpu workers
    */
    size_t num_gpu_workers() const;
    
    /**
    @brief wait for all pending graphs to complete
    */
    void wait_for_all();

  private:
    
    std::condition_variable _topology_cv;
    std::mutex _topology_mutex;
    std::mutex _queue_mutex;
    std::mutex _gpu_load_mtx;

    unsigned _num_topologies {0};
    
    // scheduler field
    std::vector<Worker> _cpu_workers;
    std::vector<Worker> _gpu_workers;
    std::vector<std::thread> _cpu_threads;
    std::vector<std::thread> _gpu_threads;
    std::vector<Notifier::Waiter> _cpu_waiters;
    std::vector<Notifier::Waiter> _gpu_waiters;

    WorkStealingQueue<Node*> _cpu_queue;
    WorkStealingQueue<Node*> _gpu_queue;

    std::atomic<size_t> _num_cpu_actives {0};
    std::atomic<size_t> _num_gpu_actives {0};
    std::atomic<size_t> _num_cpu_thieves {0};
    std::atomic<size_t> _num_gpu_thieves {0};
    std::atomic<bool>   _done {0};

    Notifier _cpu_notifier;
    Notifier _gpu_notifier;

    std::vector<Device> _gpu_devices;

    PerThread& _per_thread() const;

    bool _wait_for_cpu_tasks(Worker&, nstd::optional<Node*>&);
    bool _wait_for_gpu_tasks(Worker&, nstd::optional<Node*>&);
    
    void _exploit_cpu_tasks(Worker&, nstd::optional<Node*>&);
    void _explore_cpu_tasks(Worker&, nstd::optional<Node*>&);
    void _exploit_gpu_tasks(Worker&, nstd::optional<Node*>&);
    void _explore_gpu_tasks(Worker&, nstd::optional<Node*>&);
    void _spawn_cpu_workers();
    void _spawn_gpu_workers();
    void _prepare_gpus();
    void _create_streams(unsigned);
    void _remove_streams(unsigned);
    void _schedule(Node*, bool);
    void _schedule(std::vector<Node*>&);
    void _invoke(Worker&, Node*);
    void _invoke_host(Worker&, Node::Host&);
    void _invoke_copy(Worker&, Node::Copy&, int);
    void _invoke_fill(Worker&, Node::Fill&, int);
    void _invoke_span(Worker&, Node::Span&, int);
    void _invoke_kernel(Worker&, Node::Kernel&, int);
    void _increment_topology();
    void _decrement_topology();
    void _decrement_topology_and_notify();
    void _tear_down_topology(Topology*); 
    void _run_prologue(Topology*);
    void _run_epilogue(Topology*);
    void _host_epilogue(Node::Host&);
    void _copy_epilogue(Node::Copy&);
    void _fill_epilogue(Node::Fill&);
    void _span_epilogue(Node::Span&);
    void _kernel_epilogue(Node::Kernel&);

    // Kernels call this function to determine the GPU for execution
    int _assign_gpu(Node*);
};

// ----------------------------------------------------------------------------
// Executor::Worker
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Executor::Device
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Executor
// ----------------------------------------------------------------------------

// Constructor
inline Executor::Executor(unsigned N, unsigned M) : 
  _cpu_workers  {N},
  _cpu_waiters  {N},
  _cpu_notifier {_cpu_waiters},
  _gpu_workers  {M},
  _gpu_waiters  {M},
  _gpu_notifier {_gpu_waiters}, 
  _gpu_devices  {M} {

  // invalid number
  auto num_gpus = cuda::num_devices();
  HF_THROW_IF(num_gpus < M, "max device count is ", num_gpus);

  // prepare_gpus
  _prepare_gpus();

  // set up the workers
  _spawn_cpu_workers();
  _spawn_gpu_workers();
}

// Destructor
inline Executor::~Executor() {
  
  // wait for all topologies to complete
  wait_for_all();
  
  // shut down the scheduler
  _done = true;
  _cpu_notifier.notify(true);
  _gpu_notifier.notify(true);
  
  for(auto& t : _cpu_threads){
    t.join();
  } 

  for(auto& t : _gpu_threads){
    t.join();
  }
}

// Function: num_workers
inline size_t Executor::num_workers() const {
  return _cpu_workers.size() + _gpu_workers.size();
}

// Function: num_cpu_workers
inline size_t Executor::num_cpu_workers() const {
  return _cpu_workers.size();
}

// Function: num_gpu_workers
inline size_t Executor::num_gpu_workers() const {
  return _gpu_workers.size();
}

// Function: _per_thread
inline Executor::PerThread& Executor::_per_thread() const {
  thread_local PerThread pt;
  return pt;
}

// Procedure: _prepare_gpus
inline void Executor::_prepare_gpus() {
  for(size_t i=0; i<_gpu_devices.size(); ++i) {
    _gpu_devices[i].id = i;
    _gpu_devices[i].kstreams.resize(num_gpu_workers());
    _gpu_devices[i].cstreams.resize(num_gpu_workers());
  }
}

// Procedure: _create_streams
inline void Executor::_create_streams(unsigned i) {      

  // per-thread gpu storage
  //w.streams.resize(_gpu_devices.size());
  //w.events .resize(_gpu_devices.size());

  for(unsigned d=0; d<_gpu_devices.size(); ++d) {
    HF_WITH_CUDA_CTX(_gpu_devices[d].id) {
      HF_CHECK_CUDA(cudaStreamCreate(&_gpu_devices[d].cstreams[i]), 
        "failed to create copy stream ", i, " on device ", d
      );
      HF_CHECK_CUDA(cudaStreamCreate(&_gpu_devices[d].kstreams[i]), 
        "failed to create kernel stream ", i, " on device ", d
      );
    }
  }
}

// Procedure: _remove_streams
inline void Executor::_remove_streams(unsigned i) {

  for(unsigned d=0; d<_gpu_devices.size(); ++d) {
    HF_WITH_CUDA_CTX(_gpu_devices[d].id) {
      HF_CHECK_CUDA(cudaStreamSynchronize(_gpu_devices[d].kstreams[i]), 
        "failed to sync kernel stream ", i, " on device ", d
      );
      HF_CHECK_CUDA(cudaStreamDestroy(_gpu_devices[d].kstreams[i]), 
        "failed to destroy kernel stream ", i, " on device ", d
      );
      HF_CHECK_CUDA(cudaStreamSynchronize(_gpu_devices[d].cstreams[i]), 
        "failed to sync copy stream ", i, " on device ", d
      );
      HF_CHECK_CUDA(cudaStreamDestroy(_gpu_devices[d].cstreams[i]), 
        "failed to destroy copy stream ", i, " on device ", d
      );
    }
  }
}


// Procedure: _invoke_host
inline void Executor::_invoke_host(Worker&, Node::Host& h) {
  if(!h.work) {
    return;
  }
  h.work();
}

// Procedure: _invoke_copy
inline void Executor::_invoke_copy(Worker& worker, Node::Copy& h, int d) {
  if(!h.work) {
    return;
  }
  //auto d = h.source->_span_handle().device;
  auto s = _gpu_devices[d].cstreams[worker.id];
  //auto e = _gpu_workers[me].events[d];

  HF_WITH_CUDA_CTX(_gpu_devices[d].id) {
    //HF_CHECK_CUDA(cudaEventRecord(e, s),
    //  "failed to record event ", me, " on device ", d
    //);
    h.work(s);
    //HF_CHECK_CUDA(cudaStreamWaitEvent(s, e, 0),
    //  "failed to sync event ", me, " on device ", d
    //);
    HF_CHECK_CUDA(cudaStreamSynchronize(s),
      "failed to sync stream ", worker.id, " on device ", d
    );
  }
}

// Procedure: _invoke_span
inline void Executor::_invoke_span(Worker& worker, Node::Span& h, int d) {

  if(!h.work) {
    return;
  }
  //auto d = h.device;
  //auto e = _gpu_workers[me].events[d];
  auto s = _gpu_devices[d].cstreams[worker.id];

  HF_WITH_CUDA_CTX(_gpu_devices[d].id) {
    //HF_CHECK_CUDA(cudaEventRecord(e, s),
    //  "failed to record event ", me, " on device ", d
    //);
    h.work(_gpu_devices[d].allocator, s);
    //HF_CHECK_CUDA(cudaStreamWaitEvent(s, e, 0),
    //  "failed to sync event ", me, " on device ", d
    //);
    HF_CHECK_CUDA(cudaStreamSynchronize(s),
      "worker ", worker.id, " failed to sync stream on device ", d
    );
  }
}

// Procedure: _invoke_kernel
inline void Executor::_invoke_kernel(Worker& worker, Node::Kernel& h, int d) {
  if(!h.work) {
    return;
  }
  //auto d = h.device;
  auto s = _gpu_devices[d].kstreams[worker.id];
  //auto e = _gpu_workers[me].events[d];

  HF_WITH_CUDA_CTX(_gpu_devices[d].id) {
    //HF_CHECK_CUDA(cudaEventRecord(e, s),
    //  "failed to record event ", me, " on device ", d
    //);
    h.work(s);
    //HF_CHECK_CUDA(cudaStreamWaitEvent(s, e, 0),
    //  "failed to sync event ", me, " on device ", d
    //);
    HF_CHECK_CUDA(cudaStreamSynchronize(s),
      "worker ", worker.id, " failed to sync stream on device ", d
    );
  }
}

// Procedure: _invoke_fill
inline void Executor::_invoke_fill(Worker& worker, Node::Fill& h, int d) {
  if(!h.work) {
    return;
  }
  //auto d = h.source->_span_handle().device;
  auto s = _gpu_devices[d].cstreams[worker.id];
  //auto e = _gpu_workers[me].events[d];
  HF_WITH_CUDA_CTX(_gpu_devices[d].id) {
    //HF_CHECK_CUDA(cudaEventRecord(e, s),
    //  "failed to record event ", me, " on device ", d
    //);
    h.work(s);
    //HF_CHECK_CUDA(cudaStreamWaitEvent(s, e, 0),
    //  "failed to sync event ", me, " on device ", d
    //);
    HF_CHECK_CUDA(cudaStreamSynchronize(s),
      "failed to sync stream ", worker.id, " on device ", d
    );
  }
}


// Procedure: _assign_gpu
inline int Executor::_assign_gpu(Node* node) {

  auto &gpu_id = node->_group->device_id;
  auto id = gpu_id.load(std::memory_order_relaxed); 

  if(id == -1) {
    auto root = node->_root();
    unsigned min_load_gpu = 0;
    {
      // Only allow one worker update the load table at a time
      std::lock_guard<std::mutex> lock(_gpu_load_mtx);
      id = gpu_id.load(std::memory_order_relaxed); 
      if(id != -1) {
        return id;
      }
      int min_load = _gpu_devices[0].load.load(std::memory_order_relaxed);
      if(min_load != 0) {
        for(unsigned i=1; i<_gpu_workers.size(); i++) {
          auto load = _gpu_devices[i].load.load(std::memory_order_relaxed);
          if(load == 0) {
            gpu_id = i;
            _gpu_devices[i].load.fetch_add(root->_tree_size, std::memory_order_relaxed);
            return i;
          }

          if(load < min_load) {
            min_load = load;
            min_load_gpu = i;
          }   
        }   
      }
      gpu_id = min_load_gpu;
      _gpu_devices[min_load_gpu].load.fetch_add(root->_tree_size, std::memory_order_relaxed);
    }
    return min_load_gpu;
  }
  return id;
}

 
// Procedure: _invoke
inline void Executor::_invoke(Worker& worker, Node* node) {

  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology removal.
  const auto num_successors = node->num_successors();
  
  // Invoke the work at the node 
  struct visitor {
    Executor& e;
    Worker& worker;
    Node *node;

    void operator () (Node::Host& h) { e._invoke_host(worker, h);   }
    void operator () (Node::Copy& h) { 
      auto d = h.span->_group->device_id.load();
      assert(d != -1);
      e._invoke_copy(worker, h, d);
      //e._gpu_devices[d].load.fetch_sub(1, std::memory_order_relaxed);
    }
    void operator () (Node::Span& h) { 
      auto d = e._assign_gpu(node);
      assert(d != -1);
      h.device = d;
      e._invoke_span(worker, h, d);   
      auto remain = node->_group->num_tasks.fetch_sub(1);
      if(remain == 1) {
        auto root = node->_root();
        {
          std::lock_guard<std::mutex> lock(e._gpu_load_mtx);
          e._gpu_devices[d].load.fetch_sub(root->_tree_size, std::memory_order_relaxed);
        }
        // Recover for next iteration
        node->_group->num_tasks = root->_tree_size;
      }
    }
    void operator () (Node::Kernel& h) { 
      auto d = e._assign_gpu(node);
      assert(d != -1);
      h.device = d;
      e._invoke_kernel(worker, h, d); 
      auto remain = node->_group->num_tasks.fetch_sub(1);
      if(remain == 1) {
        auto root = node->_root();
        {
          std::lock_guard<std::mutex> lock(e._gpu_load_mtx);
          e._gpu_devices[d].load.fetch_sub(root->_tree_size, std::memory_order_relaxed);
        }
        // Recover for next iteration
        node->_group->num_tasks = root->_tree_size;
      }
    }
    void operator () (Node::Fill& h) { 
      auto d = h.span->_group->device_id.load();
      assert(d != -1);
      e._invoke_fill(worker, h, d);
      //e._gpu_devices[d].load.fetch_sub(1, std::memory_order_relaxed);
    }
  };
  
  nstd::visit(visitor{*this, worker, node}, node->_handle);
  
  // recover the runtime data  
  node->_num_dependents = static_cast<int>(node->_dependents.size());
  
  // At this point, the node storage might be destructed.
  Node* cache {nullptr};

  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_num_dependents) == 0) {
      if(node->_successors[i]->domain() != worker.domain) {
        _schedule(node->_successors[i], false);
      }
      else {
        if(cache) {
          _schedule(cache, false);
        }
        cache = node->_successors[i];
      }
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

  assert(_cpu_workers.size() != 0);
  
  // caller is a worker to this pool
  auto worker = _per_thread().worker;

  if(worker != nullptr) {
    if(!bypass) {
      if(node->domain() == Domain::CPU) {
        worker->cpu_queue.push(node);

        if(worker->domain == Domain::GPU) {
          // Make sure a CPU worker is awake
          if(_num_cpu_actives == 0 && _num_cpu_thieves == 0) {
            _cpu_notifier.notify(false);
          }
        }
      }
      else {
        worker->gpu_queue.push(node);

        if(worker->domain != Domain::GPU) {
          // Make sure a GPU worker is awake
          if(_num_gpu_actives == 0 && _num_gpu_thieves == 0) {
            _gpu_notifier.notify(false);
          }
        }
      }
    }
    else {
      assert(!worker->cache);
      worker->cache = node;
    }
    return;
  }

  // other threads
  {
    std::lock_guard<std::mutex> lock(_queue_mutex);
    if(node->domain() == Domain::CPU) {
      _cpu_queue.push(node);
    } 
    else {
      _gpu_queue.push(node);
    }
  }

  if(node->domain() == Domain::CPU) {
    _cpu_notifier.notify(false);
  }
  else {
    _gpu_notifier.notify(false);
  }
}

// Procedure: _schedule
// The main procedure to schedule a set of task nodes.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(std::vector<Node*>& nodes) {

  assert(_cpu_workers.size() != 0);
  
  // We need to cacth the node count to avoid accessing the nodes
  // vector while the parent topology is removed!
  const auto num_nodes = nodes.size();
  
  if(num_nodes == 0) {
    return;
  }

  // worker thread
  auto worker = _per_thread().worker;

  if(worker != nullptr) {
    bool spawn_gpu_task {false};
    bool spawn_cpu_task {false};
    for(size_t i=0; i<num_nodes; ++i) {
      if(nodes[i]->domain() == Domain::CPU) {
        worker->cpu_queue.push(nodes[i]);
        spawn_cpu_task = true;
      }
      else {
        worker->gpu_queue.push(nodes[i]);
        spawn_gpu_task = true;
      }
    }
    if(spawn_gpu_task && worker->domain != Domain::GPU) {
      if(_num_gpu_actives == 0 && _num_gpu_thieves == 0) {
        _gpu_notifier.notify(false);
      }
    }
    if(spawn_cpu_task && worker->domain != Domain::CPU) {
      if(_num_cpu_actives == 0 && _num_cpu_thieves == 0) {
        _cpu_notifier.notify(false);
      }
    }
    return;
  }
  
  size_t num_gpu_tasks {0};
  size_t num_cpu_tasks {0};
  // other threads
  {
    std::lock_guard<std::mutex> lock(_queue_mutex);
    for(size_t k=0; k<num_nodes; ++k) {
      if(nodes[k]->domain() == Domain::CPU) {
        _cpu_queue.push(nodes[k]);
        num_cpu_tasks ++;
      }
      else {
        _gpu_queue.push(nodes[k]);
        num_gpu_tasks ++;
      }
    }
  }

  if(num_cpu_tasks >= _cpu_workers.size()) {
    _cpu_notifier.notify(true);
  } 
  else {
    for(size_t i=0; i<num_cpu_tasks; i++) {
      _cpu_notifier.notify(false);
    }
  }
  if(num_gpu_tasks >= _gpu_workers.size()) {
    _gpu_notifier.notify(true);
  }
  else {
    for(size_t k=0; k<num_gpu_tasks; ++k) {
      _gpu_notifier.notify(false);
    }
  }
}

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Topology* tpg) {

  auto &f = tpg->_heteroflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(!tpg->_pred()) {
    tpg->_num_sinks = tpg->_cached_num_sinks;
    _schedule(tpg->_sources); 
  }
  // case 2: the final run of this topology
  else {

    // ramp up the topology
    _run_epilogue(tpg);
    
    // ready to erase the topology
    f._mtx.lock();

    // If there is another run (interleave between lock)
    if(f._topologies.size() > 1) {

      // Set the promise
      tpg->_promise.set_value();
      f._topologies.pop_front();
      f._mtx.unlock();
      
      // decrement the topology but since this is not the last we don't notify
      _decrement_topology();

      _run_prologue(&f._topologies.front());
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

  assert(_cpu_workers.size() > 0);

  _increment_topology();

  // Special case:
  //   - empty graph
  //   - pred evaluates to truth
  if(f.empty() || pred()) {
    std::promise<void> promise;
    promise.set_value();
    _decrement_topology_and_notify();
    return promise.get_future();
  }

  // TODO: throw if no workers
 
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
    }
  }
  
  // Notice here calling schedule may cause the topology to be removed sonner 
  // before the function leaves.
  if(run_now) {
    _run_prologue(tpg);
    _schedule(tpg->_sources);
  }

  return future;
}

// Procedure: _run_prologue
inline void Executor::_run_prologue(Topology* tpg) {
  
  tpg->_num_sinks = 0;
  tpg->_sources.clear();

  auto& graph = tpg->_heteroflow._graph;

  std::vector<Node*> sources;
   
  // scan each node in the graph and build up the links
  for(auto& node : graph) {

    node->_topology = tpg;

    if(node->num_dependents() == 0) {
      tpg->_sources.push_back(node.get());
    }

    if(node->num_successors() == 0) {
      tpg->_num_sinks++;
    }

    if(node->is_kernel()) {
      auto& h = node->_kernel_handle();
      assert(h.device == -1);
      for(auto s : h.sources) {
        node->_union(s);
      }
      sources.push_back(node.get());
    }
    else if(node->is_span()) {
      sources.push_back(node.get());
    }
  }

  // Constructr root device group and assign kernel node's device group
  for(auto &n: sources) {
    auto root = n->_root();
    if(root->_group == nullptr) {
      root->_group = new Node::DeviceGroup;
      root->_group->num_tasks = root->_tree_size;
    }

    if(n->_group == nullptr) {
      n->_group = root->_group;
    }
  }
  
  tpg->_cached_num_sinks = tpg->_num_sinks;
}

// Procedure: _host_epilogue
inline void Executor::_host_epilogue(Node::Host&) {
}

// Procedure: _copy_epilogue
inline void Executor::_copy_epilogue(Node::Copy&) {
}

// Procedure: _fill_epilogue
inline void Executor::_fill_epilogue(Node::Fill&) {
}

// Procedure: _span_epilogue
inline void Executor::_span_epilogue(Node::Span& h) {
  assert(h.device != -1);
  HF_WITH_CUDA_CTX(h.device) {
    _gpu_devices[h.device].allocator.deallocate(h.d_data);
  }
  h.device = -1;
  h.d_data = nullptr;
  h.d_size = 0;
}

// Procedure: _kernel_epilogue
inline void Executor::_kernel_epilogue(Node::Kernel& h) {
  assert(h.device != -1);
  h.device = -1;
}

// Procedure: _run_epilogue
inline void Executor::_run_epilogue(Topology* tpg) {
    
  if(tpg->_call != nullptr) {
    tpg->_call();
  }

  struct visitor {
    Executor& e;
    void operator () (Node::Host& h)   { e._host_epilogue(h);   }
    void operator () (Node::Copy& h)   { e._copy_epilogue(h);   }
    void operator () (Node::Span& h)   { e._span_epilogue(h);   }
    void operator () (Node::Kernel& h) { e._kernel_epilogue(h); }
    void operator () (Node::Fill& h)  { e._fill_epilogue(h);}
  };

  auto& graph = tpg->_heteroflow._graph;

  // Reset device group first
  for(auto& node : graph) {
    auto root = node->_root();
    if(root->_group != nullptr) {
      delete root->_group;
      root->_group = nullptr;
    }
    node->_group = nullptr;
  //}

  //for(auto& node : graph) {
    node->_parent = node.get();;
    node->_tree_size = 1;
    nstd::visit(visitor{*this}, node->_handle);
  }
}

// ----------------------------------------------------------------------------
// CPU workers managements
// ----------------------------------------------------------------------------

// Procedure: _spawn_cpu_workers
inline void Executor::_spawn_cpu_workers() {

  for(unsigned i=0; i<_cpu_workers.size(); ++i) {

    _cpu_workers[i].id = i;
    _cpu_workers[i].domain = Domain::CPU;

    _cpu_threads.emplace_back([this] (Worker& worker) -> void {

      // per-thread storage
      PerThread& pt = _per_thread();  
      pt.worker = &worker;
      
      nstd::optional<Node*> t;
      
      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        _exploit_cpu_tasks(worker, t);

        // wait for tasks
        if(_wait_for_cpu_tasks(worker, t) == false) {
          break;
        }
      }
      
    }, std::ref(_cpu_workers[i]));     
  }
}



// Procedure: _exploit_task
inline void Executor::_exploit_cpu_tasks(Worker& worker, nstd::optional<Node*>& t) {
  
  assert(!worker.cache);

  if(t) {
    if(_num_cpu_actives.fetch_add(1) == 0 && _num_cpu_thieves == 0) {
      _cpu_notifier.notify(false);
    }
    do {

      _invoke(worker, *t);

      if(worker.cache) {
        t = worker.cache;
        worker.cache = nullptr;
      }
      else {
        t = worker.cpu_queue.pop();
      }

    } while(t);
    _num_cpu_actives.fetch_sub(1);
  }
}

// Function: _explore_cpu_tasks
inline void Executor::_explore_cpu_tasks(Worker& thief, nstd::optional<Node*>& t) {
  
  assert(!t);

  auto W = num_workers();

  const unsigned l = 0;
  const unsigned r = W - 1;

  const size_t F = (W + 1) << 1;
  const size_t Y = 100;

  size_t f = 0;
  size_t y = 0;

  // explore
  while(!_done) {
  
    unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(thief.rdgen);
      
    // Steal cpu workers
    if(vtm < _cpu_workers.size()) {
      t = (vtm == thief.id) ? _cpu_queue.steal() : _cpu_workers[vtm].cpu_queue.steal();
    }
    else {
      t = _gpu_workers[vtm - _cpu_workers.size()].cpu_queue.steal(); 
    }

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

// Function: _wait_for_cpu_tasks
inline bool Executor::_wait_for_cpu_tasks(
  Worker& worker, nstd::optional<Node*>& t
) {

  wait_for_task:

  assert(!t);

  ++_num_cpu_thieves;

  explore_task:

  _explore_cpu_tasks(worker, t);

  if(t) {
    if(_num_cpu_thieves.fetch_sub(1) == 1) {
      _cpu_notifier.notify(false);
    }
    return true;
  }

  _cpu_notifier.prepare_wait(&_cpu_waiters[worker.id]);
  
  if(!_cpu_queue.empty()) {

    _cpu_notifier.cancel_wait(&_cpu_waiters[worker.id]);
    
    t = _cpu_queue.steal();
    if(t) {
      if(_num_cpu_thieves.fetch_sub(1) == 1) {
        _cpu_notifier.notify(false);
      }
      return true;
    }
    else {
      goto explore_task;
    }
  }

  if(_done) {
    _cpu_notifier.cancel_wait(&_cpu_waiters[worker.id]);
    _cpu_notifier.notify(true);
    _gpu_notifier.notify(true);
    --_num_cpu_thieves;
    return false;
  }

  if(_num_cpu_thieves.fetch_sub(1) == 1) {
    if(_num_cpu_actives) {
      _cpu_notifier.cancel_wait(&_cpu_waiters[worker.id]);
      goto wait_for_task;
    }
    // Check GPU workers' queues
    for(unsigned i=0; i<_gpu_workers.size(); i++) {
      if(!_gpu_workers[i].cpu_queue.empty()) {
        _cpu_notifier.cancel_wait(&_cpu_waiters[worker.id]);
        goto wait_for_task;
      }
    }
  }

  // Now I really need to relinguish my self to others
  _cpu_notifier.commit_wait(&_cpu_waiters[worker.id]);

  return true;
}

// ----------------------------------------------------------------------------
// GPU worker management
// ----------------------------------------------------------------------------

// Procedure: _spawn_gpu_workers
inline void Executor::_spawn_gpu_workers() {

  for(unsigned i=0; i<_gpu_workers.size(); ++i) {

    _gpu_workers[i].id = i;
    _gpu_workers[i].domain = Domain::GPU;

    _gpu_threads.emplace_back([this] (Worker& worker) -> void {

      // per-thread storage
      PerThread& pt = _per_thread();  
      pt.worker = &worker;

      // create gpu streams
      _create_streams(worker.id);
          
      nstd::optional<Node*> t;
      
      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        _exploit_gpu_tasks(worker, t);

        // wait for tasks
        if(_wait_for_gpu_tasks(worker, t) == false) {
          break;
        }
      }    

      // clear gpu storages
      _remove_streams(worker.id);

    }, std::ref(_gpu_workers[i]));     
  }
}

// Procedure: _exploit_gpu_tasks
inline void Executor::_exploit_gpu_tasks(
  Worker& worker, nstd::optional<Node*>& t
) {
  
  assert(!worker.cache);

  if(t) {
    if(_num_gpu_actives.fetch_add(1) == 0 && _num_gpu_thieves == 0) {
      _gpu_notifier.notify(false);
    }
    do {

      _invoke(worker, *t);

      if(worker.cache) {
        t = worker.cache;
        worker.cache = nullptr;
      }
      else {
        t = worker.gpu_queue.pop();
      }

    } while(t);

    _num_gpu_actives.fetch_sub(1);
  }
}


// Function: _wait_for_gpu_tasks
inline bool Executor::_wait_for_gpu_tasks(
  Worker& worker, nstd::optional<Node*>& t
) {

  wait_for_task:

  assert(!t);

  ++_num_gpu_thieves;

  explore_task:

  _explore_gpu_tasks(worker, t);

  if(t) {
    if(_num_gpu_thieves.fetch_sub(1) == 1) {
      _gpu_notifier.notify(false);
    }
    return true;
  }

  _gpu_notifier.prepare_wait(&_gpu_waiters[worker.id]);
  
  if(!_gpu_queue.empty()) {
    _gpu_notifier.cancel_wait(&_gpu_waiters[worker.id]);
    t = _gpu_queue.steal();
    if(t) {
      if(_num_gpu_thieves.fetch_sub(1) == 1) {
        _gpu_notifier.notify(false);
      }
      return true;
    }
    else {
      goto explore_task;
    }
  }

  if(_done) {
    _gpu_notifier.cancel_wait(&_gpu_waiters[worker.id]);
    _gpu_notifier.notify(true);
    _cpu_notifier.notify(true);
    --_num_gpu_thieves;
    return false;
  }

  if(_num_gpu_thieves.fetch_sub(1) == 1) {
    if(_num_gpu_actives) {
      _gpu_notifier.cancel_wait(&_gpu_waiters[worker.id]);
      goto wait_for_task;
    }
    // Check CPU workers' queues
    for(unsigned i=0; i<_cpu_workers.size(); i++) {
      if(!_cpu_workers[i].gpu_queue.empty()) {
        _gpu_notifier.cancel_wait(&_gpu_waiters[worker.id]);
        goto wait_for_task;
      }
    }
  }
    
  // Now I really need to relinguish my self to others
  _gpu_notifier.commit_wait(&_gpu_waiters[worker.id]);

  return true;
}

// Function: _explore_gpu_tasks
inline void Executor::_explore_gpu_tasks(
  Worker& thief, nstd::optional<Node*>& t
) {
  
  assert(!t);

  auto W = num_workers();

  const unsigned l = 0;
  const unsigned r = W - 1;

  const size_t F = (W + 1) << 1;
  const size_t Y = 100;

  size_t f = 0;
  size_t y = 0;

  // explore
  while(!_done) {
  
    unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(thief.rdgen);

    if(vtm < _cpu_workers.size()) {
      t = _cpu_workers[vtm].gpu_queue.steal(); 
    }
    else {
      vtm -= _cpu_workers.size();
      t = (vtm == thief.id) ? _gpu_queue.steal() : _gpu_workers[vtm].gpu_queue.steal();
    }
         
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


}  // end of namespace hf -----------------------------------------------------


