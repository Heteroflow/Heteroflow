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
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
  };
    
  struct PerThread {
    Executor* pool {nullptr}; 
    int worker_id  {-1};
  };
  
  struct Device {
    cuda::Allocator allocator;
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

    bool _wait_for_task(unsigned me, nstd::optional<Node*>&);

    void _spawn();
    void _create_streams(unsigned);
    void _remove_streams(unsigned);
    void _exploit_task(unsigned, nstd::optional<Node*>&);
    void _explore_task(unsigned, nstd::optional<Node*>&);
    void _schedule(Node*, bool);
    void _schedule(std::vector<Node*>&);
    void _invoke(unsigned, Node*);
    void _invoke_host(unsigned, Node::Host&);
    void _invoke_push(unsigned, Node::Push&);
    void _invoke_pull(unsigned, Node::Pull&);
    void _invoke_kernel(unsigned, Node::Kernel&);
    void _increment_topology();
    void _decrement_topology();
    void _decrement_topology_and_notify();
    void _tear_down_topology(Topology*); 
    void _run_prologue(Topology*);
    void _run_epilogue(Topology*);
    void _host_epilogue(Node::Host&);
    void _push_epilogue(Node::Push&);
    void _pull_epilogue(Node::Pull&);
    void _kernel_epilogue(Node::Kernel&);
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
  _workers  {N},
  _waiters  {N},
  _notifier {_waiters} {

  // invalid number
  auto num_devices = cuda::num_devices();
  HF_THROW_IF(num_devices < M, "max device count is ", num_devices);

  // set up the devices
  _devices.resize(M);

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

// Procedure: _create_streams
inline void Executor::_create_streams(unsigned i) {      

  auto& w = _workers[i];

  // per-thread gpu storage
  w.streams.resize(_devices.size());
  w.events .resize(_devices.size());

  for(unsigned d=0; d<_devices.size(); ++d) {
    HF_WITH_CUDA_CTX(d) {
      HF_CHECK_CUDA(cudaStreamCreate(&w.streams[d]), 
        "failed to create stream ", i, " on device ", d
      );
      HF_CHECK_CUDA(cudaEventCreate(&w.events[d]),
        "failed to create event ", i, " on device ", d
      );
    }
  }
}

// Procedure: _remove_streams
inline void Executor::_remove_streams(unsigned i) {

  auto& w = _workers[i];

  for(unsigned d=0; d<_devices.size(); ++d) {
    HF_WITH_CUDA_CTX(d) {
      HF_CHECK_CUDA(cudaStreamSynchronize(w.streams[d]), 
        "failed to sync stream ", i, " on device ", d
      );
      HF_CHECK_CUDA(cudaStreamDestroy(w.streams[d]), 
        "failed to destroy stream ", i, " on device ", d
      );
      HF_CHECK_CUDA(cudaEventDestroy(w.events[d]),
        "failed to destroy event ", i, " on device ", d
      );
    }
  }
}

// Procedure: _spawn
inline void Executor::_spawn() {
  
  for(unsigned i=0; i<_workers.size(); ++i) {
    _threads.emplace_back([this] (unsigned i) -> void {

      // per-thread storage
      PerThread& pt = _per_thread();  
      pt.pool = this;
      pt.worker_id = i;
      
      // create gpu streams
      _create_streams(i);
     
      nstd::optional<Node*> t;
      
      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        _exploit_task(i, t);

        // wait for tasks
        if(_wait_for_task(i, t) == false) {
          break;
        }
      }
      
      // clear gpu storages
      _remove_streams(i);
      
    }, i);     
  }
}

// Procedure: _exploit_task
inline void Executor::_exploit_task(unsigned i, nstd::optional<Node*>& t) {
  
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
inline void Executor::_explore_task(unsigned thief, nstd::optional<Node*>& t) {
  
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
inline bool Executor::_wait_for_task(unsigned me, nstd::optional<Node*>& t) {

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

// Procedure: _invoke_host
inline void Executor::_invoke_host(unsigned me, Node::Host& h) {
  h.work();
}

// Procedure: _invoke_push
inline void Executor::_invoke_push(unsigned me, Node::Push& h) {

  auto d = h.source->_pull_handle().device;
  auto s = _workers[me].streams[d];
  auto e = _workers[me].events[d];

  HF_WITH_CUDA_CTX(d) {
    HF_CHECK_CUDA(cudaEventRecord(e, s),
      "failed to record event ", me, " on device ", d
    );
    h.work(s);
    HF_CHECK_CUDA(cudaStreamWaitEvent(s, e, 0),
      "failed to sync event ", me, " on device ", d
    );
  }
}

// Procedure: _invoke_pull
inline void Executor::_invoke_pull(unsigned me, Node::Pull& h) {

  auto d = h.device;
  auto s = _workers[me].streams[d];
  auto e = _workers[me].events[d];

  HF_WITH_CUDA_CTX(d) {
    HF_CHECK_CUDA(cudaEventRecord(e, s),
      "failed to record event ", me, " on device ", d
    );
    h.work(_devices[d].allocator, s);
    HF_CHECK_CUDA(cudaStreamWaitEvent(s, e, 0),
      "failed to sync event ", me, " on device ", d
    );
  }
}

// Procedure: _invoke_kernel
inline void Executor::_invoke_kernel(unsigned me, Node::Kernel& h) {

  auto d = h.device;
  auto s = _workers[me].streams[d];
  auto e = _workers[me].events[d];

  HF_WITH_CUDA_CTX(d) {
    HF_CHECK_CUDA(cudaEventRecord(e, s),
      "failed to record event ", me, " on device ", d
    );
    h.work(_workers[me].streams[d]);
    HF_CHECK_CUDA(cudaStreamWaitEvent(s, e, 0),
      "failed to sync event ", me, " on device ", d
    );
  }
}
 
// Procedure: _invoke
inline void Executor::_invoke(unsigned me, Node* node) {

  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology removal.
  const auto num_successors = node->num_successors();
  
  // Invoke the work at the node 
  struct visitor {
    Executor& e;
    unsigned me;
    void operator () (Node::Host& h)   { e._invoke_host(me, h);   }
    void operator () (Node::Push& h)   { e._invoke_push(me, h);   }
    void operator () (Node::Pull& h)   { e._invoke_pull(me, h);   }
    void operator () (Node::Kernel& h) { e._invoke_kernel(me, h); }
  };
  
  nstd::visit(visitor{*this, me}, node->_handle);
  
  // recover the runtime data  
  node->_num_dependents = static_cast<int>(node->_dependents.size());
  
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

  assert(_workers.size() > 0);
  
  _increment_topology();

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
    }
  }

  tpg->_cached_num_sinks = tpg->_num_sinks;
  
  // gpu device assignment
  int cursor = 0;

  for(auto& node : graph) {
    if(node->is_kernel() || node->is_pull()) {
      assert(_devices.size() != 0);
      auto r = node->_root();
      auto d = r->_device();
      
      // need to assign a new gpu
      if(d == -1) {
        d = cursor++;
        if(cursor == _devices.size()) {
          cursor = 0;
        }
        r->_device(d);
      }
      node->_device(d);
      
      //std::cout << node->_name << " grouped to "
      //          << node->_root()->_name << " with device "
      //          << d << std::endl;
    }
  }
}

// Procedure: _host_epilogue
inline void Executor::_host_epilogue(Node::Host& h) {
}

// Procedure: _push_epilogue
inline void Executor::_push_epilogue(Node::Push& h) {
}

// Procedure: _pull_epilogue
inline void Executor::_pull_epilogue(Node::Pull& h) {
  assert(h.device != -1);
  HF_WITH_CUDA_CTX(h.device) {
    _devices[h.device].allocator.deallocate(h.d_data);
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
    void operator () (Node::Push& h)   { e._push_epilogue(h);   }
    void operator () (Node::Pull& h)   { e._pull_epilogue(h);   }
    void operator () (Node::Kernel& h) { e._kernel_epilogue(h); }
  };

  auto& graph = tpg->_heteroflow._graph;

  for(auto& node : graph) {
    nstd::visit(visitor{*this}, node->_handle);
    node->_parent = node.get();;
    node->_height = 0;
  }
}

}  // end of namespace hf -----------------------------------------------------




