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
    WorkStealingQueue<Node*> cpu_queue;
    WorkStealingQueue<Node*> gpu_queue;

    Node* cache {nullptr};
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
  };
    
  struct PerThread {
    Executor* pool {nullptr}; 
    int worker_id  {-1};
    bool gpu_thread {false};
  };
  
  struct Device {
    cuda::Allocator allocator;
		std::atomic<int> load {0};
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

    //unsigned _num_topologies {0};
		std::atomic<unsigned> _num_topologies {0};
    
    // scheduler field
    std::vector<Worker> _cpu_workers;
    std::vector<Notifier::Waiter> _cpu_waiters;
    std::vector<std::thread> _cpu_threads;

    std::vector<Worker> _gpu_workers;
    std::vector<Notifier::Waiter> _gpu_waiters;
    std::vector<std::thread> _gpu_threads;

    WorkStealingQueue<Node*> _cpu_queue;
    WorkStealingQueue<Node*> _gpu_queue;

    //std::atomic<size_t> _num_actives {0};
    std::atomic<size_t> _num_cpu_thieves {0};
    std::atomic<size_t> _num_gpu_thieves {0};
    std::atomic<bool>   _done        {0};

    Notifier _cpu_notifier;
    Notifier _gpu_notifier;

    std::vector<Device> _devices;
    
    PerThread& _per_thread() const;

    bool _wait_for_cpu_task(unsigned me, nstd::optional<Node*>&);
    void _exploit_cpu_task(unsigned, nstd::optional<Node*>&);
    void _explore_cpu_task(unsigned, nstd::optional<Node*>&);

    bool _wait_for_gpu_task(unsigned me, nstd::optional<Node*>&);
    void _exploit_gpu_task(unsigned, nstd::optional<Node*>&);
    void _explore_gpu_task(unsigned, nstd::optional<Node*>&);

    void _spawn();

    void _create_streams(unsigned);
    void _remove_streams(unsigned);
    void _schedule(Node*, bool);
    void _schedule(std::vector<Node*>&);
    void _invoke(unsigned, bool, Node*);
    void _invoke_host(unsigned, Node::Host&);
    void _invoke_push(unsigned, Node::Push&, int);
    void _invoke_pull(unsigned, Node::Pull&, int);
    void _invoke_kernel(unsigned, Node::Kernel&, int);
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

		// Kernels call this function to determine the GPU for execution
    int _assign_gpu(std::atomic<int>&);
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
	_devices {M} {

  // invalid number
  auto num_devices = cuda::num_devices();
  HF_THROW_IF(num_devices < M, "max device count is ", num_devices);

	// Must create the stream before spawning workers
  for(unsigned i=0; i<_gpu_workers.size(); ++i) {
    // create gpu streams
    _create_streams(i);
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
  _cpu_notifier.notify(true);
  _gpu_notifier.notify(true);
  
  for(auto& t : _cpu_threads){
    t.join();
  } 

  for(auto& t : _gpu_threads){
    t.join();
  }

  // Remove gpu streams after all GPU workers join
  for(unsigned i=0; i<_gpu_workers.size(); ++i) {
    _remove_streams(i);
	}
}

// Function: num_workers
inline size_t Executor::num_workers() const {
  return _cpu_workers.size() + _gpu_workers.size();
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

  auto& w = _gpu_workers[i];

  // per-thread gpu storage
  w.streams.resize(_devices.size());
  w.events .resize(_devices.size());

  for(unsigned d=0; d<_devices.size(); ++d) {
    HF_WITH_CUDA_CTX(i) {
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

  auto& w = _gpu_workers[i];

  for(unsigned d=0; d<_devices.size(); ++d) {
    HF_WITH_CUDA_CTX(i) {
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
  
  for(unsigned i=0; i<_cpu_workers.size(); ++i) {
    _cpu_threads.emplace_back([this] (unsigned i) -> void {

      // per-thread storage
      PerThread& pt = _per_thread();  
      pt.pool = this;
      pt.worker_id = i;
      
      nstd::optional<Node*> t;
      
      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        _exploit_cpu_task(i, t);

        // wait for tasks
        if(_wait_for_cpu_task(i, t) == false) {
          break;
        }
      }
      
    }, i);     
  }

  for(unsigned i=0; i<_gpu_workers.size(); ++i) {
    _gpu_threads.emplace_back([this] (unsigned i) -> void {

      // per-thread storage
      PerThread& pt = _per_thread();  
      pt.pool = this;
      pt.worker_id = i;
      pt.gpu_thread = true;
          
      nstd::optional<Node*> t;
      
      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        _exploit_gpu_task(i, t);

        // wait for tasks
        if(_wait_for_gpu_task(i, t) == false) {
          break;
        }
      }    
    }, i);     
  }
}

// Procedure: _exploit_task
inline void Executor::_exploit_cpu_task(unsigned i, nstd::optional<Node*>& t) {
  
  assert(!_cpu_workers[i].cache);

  if(t) {
    auto& worker = _cpu_workers[i];
    //if(_num_actives.fetch_add(1) == 0 && _num_thieves == 0) {
    if(_num_cpu_thieves == 0) {
      _cpu_notifier.notify(false);
    }
    do {

      _invoke(i, false, *t);

      if(worker.cache) {
        t = worker.cache;
        worker.cache = nullptr;
      }
      else {
        t = worker.cpu_queue.pop();
      }

    } while(t);

  }
}

// Function: _explore_cpu_task
inline void Executor::_explore_cpu_task(unsigned thief, nstd::optional<Node*>& t) {
  
  assert(!t);

  const unsigned l = 0;
  const unsigned r = _cpu_workers.size() + _gpu_workers.size() - 1;

  const size_t F = (_cpu_workers.size() + _gpu_workers.size() + 1) << 1;
  const size_t Y = 100;

  size_t f = 0;
  size_t y = 0;

  // explore
  while(!_done) {
  
    unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
      _cpu_workers[thief].rdgen
    );
      
    // Steal cpu workers
    if(vtm < _cpu_workers.size()) {
      t = (vtm == thief) ? _cpu_queue.steal() : _cpu_workers[vtm].cpu_queue.steal();
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

// Function: _wait_for_cpu_task
inline bool Executor::_wait_for_cpu_task(unsigned me, nstd::optional<Node*>& t) {

  wait_for_task:

  assert(!t);

  ++_num_cpu_thieves;

  explore_task:

  _explore_cpu_task(me, t);

  if(t) {
    if(_num_cpu_thieves.fetch_sub(1) == 1) {
      _cpu_notifier.notify(false);
    }
    return true;
  }

  _cpu_notifier.prepare_wait(&_cpu_waiters[me]);
  
  if(!_cpu_queue.empty()) {

    _cpu_notifier.cancel_wait(&_cpu_waiters[me]);
    
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
    _cpu_notifier.cancel_wait(&_cpu_waiters[me]);
    _cpu_notifier.notify(true);
    _gpu_notifier.notify(true);
    --_num_cpu_thieves;
    return false;
  }

  if(_num_cpu_thieves.fetch_sub(1) == 1 && _num_topologies) {
    _cpu_notifier.cancel_wait(&_cpu_waiters[me]);
    goto wait_for_task;
  }
   
  // Now I really need to relinguish my self to others
  _cpu_notifier.commit_wait(&_cpu_waiters[me]);

  return true;
}

// Procedure: _invoke_host
inline void Executor::_invoke_host(unsigned me, Node::Host& h) {
  h.work();
}



// Procedure: _invoke_push
inline void Executor::_invoke_push(unsigned me, Node::Push& h, int d) {
  //auto d = h.source->_pull_handle().device;
  //auto s = _gpu_workers[me].streams[d];
  //auto e = _gpu_workers[me].events[d];
  auto s = _gpu_workers[d].streams[me];
  auto e = _gpu_workers[d].events[me];

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
inline void Executor::_invoke_pull(unsigned me, Node::Pull& h, int d) {
  //auto d = h.device;
  //auto s = _gpu_workers[me].streams[d];
  //auto e = _gpu_workers[me].events[d];
  auto s = _gpu_workers[d].streams[me];
  auto e = _gpu_workers[d].events[me];

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
inline void Executor::_invoke_kernel(unsigned me, Node::Kernel& h, int d) {
  //auto d = h.device;
  //auto s = _gpu_workers[me].streams[d];
  //auto e = _gpu_workers[me].events[d];
  auto s = _gpu_workers[d].streams[me];
  auto e = _gpu_workers[d].events[me];

  HF_WITH_CUDA_CTX(d) {
    HF_CHECK_CUDA(cudaEventRecord(e, s),
      "failed to record event ", me, " on device ", d
    );
    //h.work(_gpu_workers[me].streams[d]);
    h.work(_gpu_workers[d].streams[me]);
    HF_CHECK_CUDA(cudaStreamWaitEvent(s, e, 0),
      "failed to sync event ", me, " on device ", d
    );
  }
}


// Procedure: _assign_gpu
inline int Executor::_assign_gpu(std::atomic<int>& gpu_id) {
  auto id = gpu_id.load(std::memory_order_relaxed); 
  if(id == -1) {
    unsigned min_load_gpu = 0;
    int min_load = _devices[0].load.load(std::memory_order_relaxed);

    for(unsigned i=1; i<_gpu_workers.size(); i++) {
			auto load = _devices[i].load.load(std::memory_order_relaxed);
      if(load < min_load) {
        min_load = load;
        min_load_gpu = i;
      }   
    }   

    if(gpu_id.compare_exchange_strong(id, min_load_gpu, std::memory_order_seq_cst, std::memory_order_relaxed)) {
			_devices[min_load_gpu].load.fetch_add(1, std::memory_order_relaxed);
      return min_load_gpu;
    }
  }
	_devices[id].load.fetch_add(1, std::memory_order_relaxed);
  return id;
}

 
// Procedure: _invoke
inline void Executor::_invoke(unsigned me, bool gpu_thread, Node* node) {

  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology removal.
  const auto num_successors = node->num_successors();
  
  // Invoke the work at the node 
  struct visitor {
    Executor& e;
		unsigned me;
    Node *node;

    void operator () (Node::Host& h)   { e._invoke_host(me, h);   }
    void operator () (Node::Push& h)   { 
      auto d = e._assign_gpu(h.source->_group->device_id);
      e._invoke_push(me, h, d);
			e._devices[d].load.fetch_sub(1, std::memory_order_relaxed);
    }
    void operator () (Node::Pull& h)   { 
      auto d = e._assign_gpu(node->_group->device_id);
			h.device = d;
      e._invoke_pull(me, h, d);   
			e._devices[d].load.fetch_sub(1, std::memory_order_relaxed);
    }
    void operator () (Node::Kernel& h) { 
      auto d = e._assign_gpu(node->_group->device_id);
			h.device = d;
      e._invoke_kernel(me, h, d); 
			e._devices[d].load.fetch_sub(1, std::memory_order_relaxed);
    }
  };
  
  nstd::visit(visitor{*this, me, node}, node->_handle);
  
  // recover the runtime data  
  node->_num_dependents = static_cast<int>(node->_dependents.size());
  
  // At this point, the node storage might be destructed.
  Node* cache {nullptr};

  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_num_dependents) == 0) {
      if(cache) {
        _schedule(cache, false);
				cache = node->_successors[i];
      }
      else {
        if((node->_successors[i]->is_host() && !gpu_thread) || 
           (!node->_successors[i]->is_host() && gpu_thread) ) {
          cache = node->_successors[i];
        }
        else {
          _schedule(node->_successors[i], false);
        }
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
  auto& pt = _per_thread();

  if(pt.pool == this) {
    auto &w = pt.gpu_thread ? _gpu_workers[pt.worker_id] : _cpu_workers[pt.worker_id];
    if(!bypass) {
      if(node->is_host()) {
        w.cpu_queue.push(node);
      }
      else {
        w.gpu_queue.push(node);
      }
    }
    else {
      assert(!w.cache);
      w.cache = node;
    }
    return;
  }

  // other threads
  {
    std::lock_guard<std::mutex> lock(_queue_mutex);
    if(node->is_host()) {
      _cpu_queue.push(node);
    } 
    else {
      _gpu_queue.push(node);
    }
  }

  if(node->is_host()) {
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
  auto& pt = _per_thread();

  if(pt.pool == this) {
    auto &w = pt.gpu_thread ? _gpu_workers[pt.worker_id] : _cpu_workers[pt.worker_id];
    for(size_t i=0; i<num_nodes; ++i) {
      if(nodes[i]->is_host()) {
        w.cpu_queue.push(nodes[i]);
      }
      else {
        w.gpu_queue.push(nodes[i]);
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
      if(nodes[k]->is_host()) {
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

		// Reset the device group
		for(auto &n: tpg->_heteroflow._graph) {
			if(n->is_device()) {
				assert(n->_group != nullptr);
				n->_group->device_id = -1;
			}
		}
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

	// Put these in topology struct ?
	std::vector<Node*> pull_nodes;
	std::vector<Node*> kernel_nodes;
	//std::vector<Node*> push_nodes;
   
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
			kernel_nodes.push_back(node.get());
    }

		if(node->is_pull()) {
			pull_nodes.push_back(node.get());
		}

		//if(node->is_push()) {
		//	push_nodes.push_back(node.get());
		//}
  }

	// Constructr root device group and assign kernel node's device group
	for(auto &n: kernel_nodes) {
    auto root = n->_root();
		if(root->_group == nullptr) {
			root->_group = new Node::DeviceGroup;
		}
		if(n->_group == nullptr) {
			n->_group = root->_group;
		}
	}

	// Set pull node's device group
	for(auto &n: pull_nodes) {
    auto root = n->_root();
		assert(root->_group != nullptr);
		if(n->_group == nullptr) {
			n->_group = root->_group;
		}
	}

	//// Set push node's device group
	//for(auto &n: push_nodes) {
	//	for(auto d: n->_dependents) {
	//		if(d->is_kernel() || d->is_pull()) {
	//			assert(d->_group != nullptr);
	//			n->_parent = d->_parent;
	//			n->_group = d->_group;
  //      break;
	//		}
	//	}
	//}

  //for(auto& node : graph) {
  //  if(node->is_push()) {
  //    for(auto d: node->_dependents) {
  //      if(d->is_kernel() || d->is_pull()) {
  //         node->_parent = d->_parent;
  //         break;
  //      }
  //    }
  //  }
  //}

  tpg->_cached_num_sinks = tpg->_num_sinks;

  
  // gpu device assignment
  //int cursor = 0;
  //for(auto& node : graph) {
  //  if(node->is_kernel() || node->is_pull()) {
  //    assert(_devices.size() != 0);
  //    auto r = node->_root();
  //    auto d = r->_device();
  //    
  //    // need to assign a new gpu
  //    if(d == -1) {
  //      d = cursor++;
  //      if(cursor == _devices.size()) {
  //        cursor = 0;
  //      }
  //      r->_device(d);
  //    }
  //    node->_device(d);
  //    
  //    //std::cout << node->_name << " grouped to "
  //    //          << node->_root()->_name << " with device "
  //    //          << d << std::endl;
  //  }
  //}
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

	// Reset device group first
  for(auto& node : graph) {
    auto root = node->_root();
		if(root->_group != nullptr) {
			delete root->_group;
			root->_group = nullptr;
		}
    node->_group = nullptr;
  }

  for(auto& node : graph) {
    nstd::visit(visitor{*this}, node->_handle);
    node->_parent = node.get();;
    node->_height = 0;
  }
}





// -------------------------  GPU worker routines -----------------------------

// Procedure: _exploit_gpu_task
inline void Executor::_exploit_gpu_task(unsigned i, nstd::optional<Node*>& t) {
  
  assert(!_gpu_workers[i].cache);

  if(t) {
    auto& worker = _gpu_workers[i];
    if(_num_gpu_thieves == 0) {
      _gpu_notifier.notify(false);
    }
    do {

      _invoke(i, true, *t);

      if(worker.cache) {
        t = worker.cache;
        worker.cache = nullptr;
      }
      else {
        t = worker.gpu_queue.pop();
      }

    } while(t);

  }
}


// Function: _wait_for_gpu_task
inline bool Executor::_wait_for_gpu_task(unsigned me, nstd::optional<Node*>& t) {

  wait_for_task:

  assert(!t);

  ++_num_gpu_thieves;

  explore_task:

  _explore_gpu_task(me, t);

  if(t) {
    if(_num_gpu_thieves.fetch_sub(1) == 1) {
      _gpu_notifier.notify(false);
    }
    return true;
  }

  _gpu_notifier.prepare_wait(&_gpu_waiters[me]);
  
  if(!_gpu_queue.empty()) {
    _gpu_notifier.cancel_wait(&_gpu_waiters[me]);
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
    _gpu_notifier.cancel_wait(&_gpu_waiters[me]);
    _gpu_notifier.notify(true);
    _cpu_notifier.notify(true);
    --_num_gpu_thieves;
    return false;
  }

  if(_num_gpu_thieves.fetch_sub(1) == 1 && _num_topologies) {
    _gpu_notifier.cancel_wait(&_gpu_waiters[me]);
    goto wait_for_task;
  }
    
  // Now I really need to relinguish my self to others
  _gpu_notifier.commit_wait(&_gpu_waiters[me]);

  return true;
}

// Function: _explore_gpu_task
inline void Executor::_explore_gpu_task(unsigned thief, nstd::optional<Node*>& t) {
  
  assert(!t);

  const unsigned l = 0;
  const unsigned r = _cpu_workers.size() + _gpu_workers.size() - 1;

  const size_t F = (_cpu_workers.size() + _gpu_workers.size() + 1) << 1;
  const size_t Y = 100;

  size_t f = 0;
  size_t y = 0;

  // explore
  while(!_done) {
  
    unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
      _gpu_workers[thief].rdgen
    );

    // Steal cpu workers
    if(vtm < _cpu_workers.size()) {
      t = _cpu_workers[vtm].gpu_queue.steal(); 
    }
    else {
      t = (vtm - _cpu_workers.size() == thief) ? _gpu_queue.steal() : _gpu_workers[vtm].gpu_queue.steal();
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


