
class TaskBase {
 
  virtual operator ()() = 0;
  
};

class HeteroVector : public TaskBase {
  
  public:
    
    template <typename T>
    HeteroTaskLineage(T*, size_t);
  
  private:

    void* _d_data {nullptr};
    void* _h_data {nullptr};

    size_t _d_N {0};
    size_t _h_N {0};

    // stream
    // device
};

using _1 = std::placeholder::_1;
using _2 = std::placeholder::_2;

thrust::host_vector<int> d_pinx;     // device 0
thrust::host_vector<int> d_piny;     // device 0
thrust::device_vector<int> h_pinx;   // device 0

h_pinx[1] ...
d_pinx = h_pinx;   // cudaMemcpy


__global__ void k1(int* data, size_t N, int v);
__global__ void k2(float v, int* data, size_t N);

// create a heteroflow
Heteroflow hf;

/*auto cpu_task = hf.emplace([&](){ 
  placement(1);
  h_pinx[i] = 1;
  h_piny[i] = 1;
});

auto px = hf.pull(d_pinx, h_pinx);
auto py = hf.pull(d_piny, h_piny);

auto k1 = hf.kernel(
  policy, compute_hpwl, 
  d_pinx.data(), d_pinx.size(),
  d_piny.data(), d_piny.size()
);

auto sx = hf.push(h_pinx, d_pinx);
auto sy = hf.push(h_piny, d_piny);

cpu_task.precede(px, py);
k1.gather(px, py);
k1.precede(sx, sy);*/

__global__ void k1(float* d_pinx, float* d_piny, size_t Nx, size_t Ny) {
  if(threadIdx....)
}

float* h_pinx, *h_piny, 
size_t n_x, n_y;

auto cpu_task1 = hf.host([](){});
auto cpu_task2 = hf.host([](){});
auto gpu_task1 = hf.pull(h_pinx, n_x);   // internally stores d_pinx, Nx
auto gpu_task2 = hf.pull(h_piny, n_y);   // internally stores d_piny, Ny

__global__ void k2 (float* pinx, size_t N, float v, size_t other_value_from_cpu);
gpu_task1.kernel(policy, k2,
  [&other_value_from_cpu] (float* d_pinx, size_t N) {
    return {d_ptr, N, 1.2, other_value_from_cpu};
  }
);

auto k1 = hf.kernel(policy, k1, 
  dx, n_x, h_piny, n_y
);
// internally finds a way to call k1

// make sure the kernel is called on the right device
cudaSetDevice(1);
k1<<< Nx/256, 256, Stream/* decided by scheduler */ >>>(d_pinx, d_piny, Nx, Ny);

auto gpu_task1_1 = hf.push(h_pinx, n_x);
auto gpu_task2_1 = hf.push(h_piny, n_y);

cpu_task1.precede(gpu_task1);
cpu_task2.precede(gpu_task2);
gpu_task1.precede(k1);
gpu_task2.precede(k1);
k1.precede(gpu_task1_1);
k2.precede(gpu_task2_1);

hf::Executor executor(2, 3);
executor.run(hf);

  //void* d_ptr;
  //size_t d_bytes;

  //// chun-xun's code
  //float* d_ptr_x;
  //float* d_ptr_y;
  //size_t N;

__global__ void compute_hpwl(int* x, size_t n_x, float* y, size_t n_y);

gpu_task.pull();    // x (int)
gpu_task2.pull();   // y (float)
gpu_task.kernel(policy, compute_hpwl, 
  [&] (std::span<int> span) -> std::tuple {
    return {span.data(), n_x, span.subspan(n_x), n_y};
  }
);
gpu_task.push();    // h_x (int)
gpu_task2.push();   // h_y (float)

//auto cpu_task  = hf.host([](){});
//auto task_pull = hf.pull(htl);
//auto task_call = hf.kernel(policy, kernel_1,
//  eigen::TensorView<int, 3, 3>(host_vector)
//
//);
//auto task_push = hf.push(htl);

// By default, we use 1D vector (similar to thrust::device_vector and
// thrust::host_vector).

// In the future, we might support different data types, for example:
// emplace_htl< eigen::Tensor<int> >(...);

// Here we can actually borrow the idea from git
gpu_task.pull();                          // pull data to device
gpu_task.launch(policy, k1, _1, _2, 19);  // launch kernel
gpu_task.launch(policy, k2, 2f, _1, _2);  // launch kernel
gpu_task.push();                          // push data to host

// At this poinet, we only have a task lineage, nothing actually get executed

// build a dependency
cpu_task.precede(gpu_task);

// run the heteroflow on an executor
Executor executor;   
executor.run(hf);


