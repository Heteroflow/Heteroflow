# Heteroflow
Heterogeneous Task Programming 

# Study note

```cpp

using _1 = std::placeholder::_1;
using _2 = std::placeholder::_2;

__global__ void k1(int* data, size_t N, int v);
__global__ void k2(float v, int* data, size_t N);

// create a heteroflow
Heteroflow hf;

auto cpu_task = hf.emplace([](){ ... });
auto gpu_task = hf.emplace_gtl<int>(host_data, N);   // GPU task lineage (GTL)

// By default, we use 1D vector (similar to thrust::device_vector and
// thrust::host_vector).

// In the future, we might support different data types, for example:
// emplace_gtl< eigen::Tensor<int> >(...);

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

```



