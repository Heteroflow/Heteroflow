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
    std::vector<Worker> _workers;
    std::vector<Notifier::Waiter> _waiters;
    std::vector<std::thread> _threads;

    std::vector<Worker> _gpu_workers;
    std::vector<Notifier::Waiter> _gpu_waiters;
    std::vector<std::thread> _gpu_threads;

    std::vector<Device> _devices;

    WorkStealingQueue<Node*> _queue;
    WorkStealingQueue<Node*> _gpu_queue;

    //std::atomic<size_t> _num_actives {0};
    std::atomic<size_t> _num_thieves {0};
    std::atomic<size_t> _num_gpu_thieves {0};
    std::atomic<bool>   _done        {0};

    Notifier _notifier;
    Notifier _gpu_notifier;
    
    PerThread& _per_thread() const;

    bool _wait_for_task(unsigned me, nstd::optional<Node*>&);
    bool _wait_for_gpu_task(unsigned me, nstd::optional<Node*>&);
    void _exploit_gpu_task(unsigned, nstd::optional<Node*>&);
    void _explore_gpu_task(unsigned, nstd::optional<Node*>&);

    void _spawn();

    void _create_streams(unsigned);
    void _remove_streams(unsigned);
    void _exploit_task(unsigned, nstd::optional<Node*>&);
    void _explore_task(unsigned, nstd::optional<Node*>&);
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

    void _set_devicegroup(Graph&);
    void _reset_devicegroup(Graph&);
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
  _notifier {_waiters},
  _gpu_workers  {M},
  _gpu_waiters  {M},
  _gpu_notifier {_gpu_waiters} {

  // invalid number
  auto num_devices = cuda::num_devices();
  HF_THROW_IF(num_devices < M, "max device count is ", num_devices);

  // set up the devices
  _devices.resize(M);

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
  _notifier.notify(true);
  _gpu_notifier.notify(true);
  
  for(auto& t : _threads){
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
  
  for(unsigned i=0; i<_workers.size(); ++i) {
    _threads.emplace_back([this] (unsigned i) -> void {

      // per-thread storage
      PerThread& pt = _per_thread();  
      pt.pool = this;
      pt.worker_id = i;
      
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
inline void Executor::_exploit_task(unsigned i, nstd::optional<Node*>& t) {
  
  assert(!_workers[i].cache);

  if(t) {
    auto& worker = _workers[i];
    //if(_num_actives.fetch_add(1) == 0 && _num_thieves == 0) {
    if(_num_thieves == 0) {
      _notifier.notify(false);
    }
    do {

      _invoke(i, false, *t);

      if(worker.cache) {
        t = worker.cache;
        worker.cache = nullptr;
      }
      else {
        t = worker.queue.pop();
      }

    } while(t);

  }
}

// Function: _explore_task
inline void Executor::_explore_task(unsigned thief, nstd::optional<Node*>& t) {
  
  assert(!t);

  const unsigned l = 0;
  const unsigned r = _workers.size() + _gpu_workers.size() - 1;

  const size_t F = (_workers.size() + _gpu_workers.size() + 1) << 1;
  const size_t Y = 100;

  size_t f = 0;
  size_t y = 0;

  // explore
  while(!_done) {
  
    unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
      _workers[thief].rdgen
    );
      
    // Steal cpu workers
    if(vtm < _workers.size()) {
      t = (vtm == thief) ? _queue.steal() : _workers[vtm].queue.steal();
    }
    else {
      t = _gpu_workers[vtm - _workers.size()].queue.steal(); 
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
    _gpu_notifier.notify(true);
    --_num_thieves;
    return false;
  }

  if(_num_thieves.fetch_sub(1) == 1 && _num_topologies) {
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
      auto d = node->_set_gpu_device(me, node->_group->device_id);
      e._invoke_push(me, h, d);
    }
    void operator () (Node::Pull& h)   { 
      auto d = node->_set_gpu_device(me, node->_group->device_id);
			h.device = d;
      e._invoke_pull(me, h, d);   
    }
    void operator () (Node::Kernel& h) { 
      auto d = node->_set_gpu_device(me, node->_group->device_id);
			h.device = d;
      e._invoke_kernel(me, h, d); 
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
  
  assert(_workers.size() != 0);
  
  // caller is a worker to this pool
  auto& pt = _per_thread();

  if(pt.pool == this) {
    auto &w = pt.gpu_thread ? _gpu_workers[pt.worker_id] : _workers[pt.worker_id];
    if(!bypass) {
      if(node->is_host()) {
        w.queue.push(node);
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
      _queue.push(node);
    } 
    else {
      _gpu_queue.push(node);
    }
  }

  if(node->is_host()) {
    _notifier.notify(false);
  }
  else {
    _gpu_notifier.notify(false);
  }
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
    auto &w = pt.gpu_thread ? _gpu_workers[pt.worker_id] : _workers[pt.worker_id];
    for(size_t i=0; i<num_nodes; ++i) {
      if(nodes[i]->is_host()) {
        w.queue.push(nodes[i]);
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
        _queue.push(nodes[k]);
        num_cpu_tasks ++;
      }
      else {
        _gpu_queue.push(nodes[k]);
        num_gpu_tasks ++;
      }
    }
  }

  if(num_cpu_tasks >= _workers.size()) {
    _notifier.notify(true);
	}	
	else {
	  for(size_t i=0; i<num_cpu_tasks; i++) {
		  _notifier.notify(false);
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


// Procedure: _set_devicegroup
inline void Executor::_set_devicegroup(Graph& graph) {

  // Find the connected kernel and set them to the same group
	for(auto& node : graph) {
		if(node->is_device() && node->_group == nullptr) {
			node->_group = new Node::DeviceGroup();
			const auto ptr = node->_group;
			std::vector<Node*> reachable {node.get()};
      while(!reachable.empty()) {
				auto n = reachable.back();
				reachable.pop_back();
				// Check successors
				for(auto &s: n->_successors) {
					if(s->is_device() && s->_group == nullptr) {
						reachable.emplace_back(s);
						s->_group = ptr;
					}
				}
        // Check depedents
				for(auto &d: n->_dependents) {
					if(d->is_device() && d->_group == nullptr) {
						reachable.emplace_back(d);
						d->_group = ptr;
					}
				}
			} // End of while-loop
		}   // End of if
	}     // End of for-loop
}

// Procedure: _reset_devicegroup
inline void Executor::_reset_devicegroup(Graph &graph) {

	// Reset the group of kernels
	for(auto& node : graph) {
		if(node->is_device() && node->_group != nullptr) {
			const auto ptr = node->_group;
			std::vector<Node*> reachable {node.get()};
			node->_group = nullptr;
      while(!reachable.empty()) {
				auto n = reachable.back();
				reachable.pop_back();
				// Check successors
				for(auto &s: n->_successors) {
					if(s->is_device() && s->_group == ptr) {
						reachable.emplace_back(s);
						s->_group = nullptr;
					}
				}
        // Check depedents
				for(auto &d: n->_dependents) {
					if(d->is_device() && d->_group == ptr) {
						reachable.emplace_back(d);
						d->_group = nullptr;
					}
				}
			} // End of while-loop
			free(ptr);
		}   // End of if
	}     // End of for-loop
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
    
    //if(node->is_kernel()) {
    //  auto& h = node->_kernel_handle();
    //  assert(h.device == -1);
    //  for(auto s : h.sources) {
    //    node->_union(s);
    //  }
    //}
  }

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

	_set_devicegroup(graph);
  
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

  for(auto& node : graph) {
    nstd::visit(visitor{*this}, node->_handle);
    //node->_parent = node.get();;
    //node->_height = 0;
  }
 	_reset_devicegroup(tpg->_heteroflow._graph);
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
    _notifier.notify(true);
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
  const unsigned r = _workers.size() + _gpu_workers.size() - 1;

  const size_t F = (_workers.size() + _gpu_workers.size() + 1) << 1;
  const size_t Y = 100;

  size_t f = 0;
  size_t y = 0;

  // explore
  while(!_done) {
  
    unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
      _gpu_workers[thief].rdgen
    );

    // Steal cpu workers
    if(vtm < _workers.size()) {
      t = _workers[vtm].gpu_queue.steal(); 
    }
    else {
      t = (vtm - _workers.size() == thief) ? _gpu_queue.steal() : _gpu_workers[vtm].gpu_queue.steal();
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


