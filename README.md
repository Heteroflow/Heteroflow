# Heteroflow <img align="right" width="10%" src="images/heteroflow-logo.png">

A header-only C++ library to help you quickly write
concurrent CPU-GPU programs using task models

:exclamation: This is a working repository with many things under construction,
but with enough information to highlight the spirit of Heteroflow.

# Why Heteroflow?

Modern high-performance computing leverages a heterogeneous mix of 
both CPU and GPU devices.
However, concurrent CPU-GPU programming is notoriously difficult
due to many implementation details.
Heteroflow helps you deal with this challenge through a new programming model
using modern C++ and Nvidia CUDA Toolkit.

# Write your First Heteroflow Program

The following example [saxpy.cu](./examples/saxpy.cu) implements
the canonical single-precision A·X Plus Y ("saxpy") operation.


```cpp
#include <heteroflow/heteroflow.hpp>  // Heteroflow is header-only

__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void) {

  const int num_items = 1<<20;                    // total items
  const int num_bytes = num_items*sizeof(float);  // total bytes
  float* x {nullptr};
  float* y {nullptr};

  hf::Executor executor;                          // create an executor
  hf::Heteroflow hf("saxpy");                     // create a task dependency graph 

  auto host_x = hf.host([&](){ x = new float[num_items]; std::fill_n(x, num_items, 1.0f); });
  auto host_y = hf.host([&](){ y = new float[num_items]; std::fill_n(y, num_items, 2.0f); });
  auto pull_x = hf.pull(std::ref(x), num_bytes); 
  auto pull_y = hf.pull(std::ref(y), num_bytes);           
  auto kernel = hf.kernel((num_items+255)/256, 256, 0, saxpy, num_items, 2.0f, pull_x, pull_y);
  auto push_x = hf.push(pull_x, std::ref(x));
  auto push_y = hf.push(pull_y, std::ref(y));

  host_x.precede(pull_x);                         // host_x runs before pull_x
  host_y.precede(pull_y);                         // host_y runs before pull_y
  kernel.precede(push_x, push_y)                  // kernel runs before push_x and push_y
        .succeed(pull_x, pull_y);                 // kernel runs after  pull_x and pull_Y

  executor.run(hf).wait();                        // execute the task dependency graph
}
```

The saxpy task dependency graph is shown in the following figure:

<img src="images/saxpy.png" width="65%">


Compile and run the code with the following commands:

```bash
~$ nvcc saxpy.cu -std=c++14 -O2 -o saxpy -I path/to/Heteroflow/header
~$ ./saxpy
```

# Create a Heteroflow Application

Heteroflow manages concurrent CPU-GPU programming 
using a *task dependency graph* model.
Each node in the graph represents a task and each edge indicates
a dependency constraint between two nodes.
Most applications are developed through the following steps:

## Step 1: Create a Heteroflow Graph

Create a heteroflow object to build a task dependency graph:

```cpp
hf::Heteroflow heteroflow;
```

Each task belongs to one of the following categories: 
*host*, *pull*, *push*, and *kernel*.


### Host Task

A host task is a callable for which [std::invoke][std::invoke] is applicable
on any CPU core.

```cpp
hf::HostTask host = heteroflow.host([](){ std::cout << "my host task\n"; });
```

### Pull Task

A pull task allocates memory on a GPU and copies
data from the host to the GPU.

```cpp
float* data = new float[10];
hf::PullTask pull = heteroflow.pull(data, 10*sizeof(float));
```

If the data pointer is given a `nullptr`, 
the pull task only allocates a GPU memory area of the given bytes.
In this case, you can give an additional value to initialize 
for each byte of the memory.

```cpp
hf::PullTask gpu_mem1 = heteroflow.pull(nullptr, 256);
hf::PullTask gpu_mem2 = heteroflow.pull(nullptr, 256, 0); 
```

The arguments to pass to `Heteroflow::pull` can be *stateful*,
in which variables are captured by reference through [std::ref][std::ref].
Stateful capture is useful for variables that cannot decide 
values until runtime,
or that take advantage of host tasks to modify values for 
task parallelism.

```cpp
float* runtime_data{nullptr};
size_t size{0}, bytes{0};
hf::HostTask host = heteroflow.host([&](){      // capture everything by reference
  runtime_data = new float[size=100];           // change size and data at runtime
  bytes = size*sizeof(float);
});
hf::PullTask stateful_pull = heteroflow.pull(std::ref(runtime_data), std::ref(bytes));
```
 
### Push Task

A push task copies GPU data associated with a pull task back to a host memory area.

```cpp
// using the pull and data example above
hf::PushTask push = heteroflow.push(pull, data, 10*sizeof(float));
```

You can give an *offset* in bytes to change the beginning point of the memory block in the pull task.

```cpp
// skips the first 3 floats in the gpu memory block of pull
hf::PushTask push = heteroflow.push(pull, data, 3*sizeof(float), 7*sizeof(float))
```

### Kernel Task

A kernel task offloads a kernel function to a GPU device.
Heteroflow abstracts GPU memory through pull tasks 
to perform automatic device mapping and memory allocation at runtime.
Having said that, 
each pull task can convert to a pointer of whatever data type specified 
in the kernel function.

```cpp
__global__ void my_kernel(float* data, int N);

dim3 grid {1, 1, 1}, block {10, 1, 1};
size_t shared_memory {0};

hf::KernelTask k1 = hf.kernel(grid, block, shared_memory, my_kernel, pull, 10); 
```

Heteroflow gives users full privileges to leverage their domain-specific knowledge
to write a high-performance [CUDA][cuda-zone] kernel. 
Users focus on developing kernels and CPU tasks using native CUDA toolkits,
while leaving task parallelism to Heteroflow.

## Step 2: Define Task Dependencies

You can add dependency links between tasks to enforce one task to run after another.
The dependency links must be a
[Directed Acyclic Graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph).
You can add a preceding link to force one task to run before another.

```cpp
A.precede(B);  // A runs before B.
```

Or you can add a succeed link to force one task to run after another.

```cpp
A.succeed(B);  // A runs after B
```

## Step 3: Execute a Heteroflow

To execute a heteroflow, you need to create an *executor*.
An executor manages a set of worker threads to execute a heteroflow
and perform automatic computation offloading to GPUs
through an efficient *work-stealing* algorithm.

```cpp
tf::Executor executor;
```

The executor provides many methods to run a heteroflow.
You can run a heteroflow one time, multiple times, or 
based on a stopping criteria.
These methods are *non-blocking* with a [std::future][std::future] return
to let you query the execution status.
All executor methods are *thread-safe*.

```cpp
std::future<void> r1 = executor.run(heteroflow);       // run the heteroflow once
std::future<void> r2 = executor.run_n(heteroflow, 2);  // run the heteroflow twice

// keep running until the predicate becomes true (4 times in this example)
executor.run_until(taskflow, [counter=4](){ return --counter == 0; } );
```

You can call `wait_for_all` to block the executor until all associated taskflows complete.

```cpp
executor.wait_for_all();  // block until all associated tasks finish
```

Notice that executor does not own any heteroflow. 
It is your responsibility to keep a heteroflow alive during its execution,
or it can result in undefined behavior.
In most applications, you need only one executor to run multiple hteroflows
each representing a specific part of your parallel decomposition.

# System Requirements

To use Heteroflow, you need a [Nvidia's CUDA Compiler (NVCC)][nvcc] 
of version at least 9.0 to support C++14 standards.

# License

Heteroflow is licensed under the [MIT License](./LICENSE).

* * *

[std::ref]:              https://en.cppreference.com/w/cpp/utility/functional/ref
[span::data]:            https://en.cppreference.com/w/cpp/container/span/data
[std::invoke]:           https://en.cppreference.com/w/cpp/utility/functional/invoke
[std::future]:           https://en.cppreference.com/w/cpp/thread/future


[cuda-zone]:             https://developer.nvidia.com/cuda-zone
[nvcc]:                  https://developer.nvidia.com/cuda-llvm-compiler

