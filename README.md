# Heteroflow <img align="right" width="10%" src="images/heteroflow-logo.png">

A header-only C++ library to help you quickly write
concurrent CPU-GPU programs

:exclamation: This is a working repository with many things under construction,
but with enough information to highlight the spirit of Heteroflow.

# Why Heteroflow?

Modern parallel applications typically contain
a broad use of CPUs and GPUs.
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

  const int N = 1<<20;
  std::vector<float> x, y;

  hf::Executor executor;              // create an executor
  hf::Heteroflow hf("saxpy");         // create a task dependency graph 

  auto host_x = hf.host([&](){ x.resize(N, 1.0f); });
  auto host_y = hf.host([&](){ y.resize(N, 2.0f); });
  auto pull_x = hf.pull(x); 
  auto pull_y = hf.pull(y);           
  auto kernel = hf.kernel(saxpy, N, 2.0f, pull_x, pull_y)
                  .grid_x((N+255)/256)
                  .block_x(256)
  auto push_x = hf.push(pull_x, x);
  auto push_y = hf.push(pull_y, y);

  host_x.precede(pull_x);             // host_x runs before pull_x
  host_y.precede(pull_y);             // host_y runs before pull_y
  kernel.precede(push_x, push_y)      // kernel runs before push_x and push_y
        .succeed(pull_x, pull_y);     // kernel runs after  pull_x and pull_Y

  executor.run(hf).wait();            // execute the task dependency graph
}
```

The saxpy task dependency graph is shown in the following figure:

<img src="images/saxpy.png" width="70%">


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

Each task belongs to one of the four categories: 
*host*, *pull*, *push*, and *kernel*.


### Host Task

A host task is a callable for which [std::invoke][std::invoke] is applicable
on any CPU core.

```cpp
hf::HostTask host = heteroflow.host([](){ std::cout << "my host task\n"; });
```

### Pull Task

A pull task manages GPU memory allocation
and copies data from the host to a GPU device.

```cpp
std::vector<int> data1(100);
hf::PullTask pull1 = heteroflow.pull(data1);      // data span from a container

float* data2 = new float[10];
hf::PullTask pull2 = heteroflow.pull(data2, 10);  // data span from a raw memory block
```

The arguments to forward to `Heteroflow::pull` 
must conform to the contract of C++20 [std::span][std::span]
through which we use [span::data][span::data] and
[span::size_bytes][span::size_bytes] 
to perform data copy.
Currently, we use the drop-in replacement [span-lite][span-lite].
 
### Push Task

A push task copies GPU data associated with a pull task back to the host.
The arguments to forward to `Heteroflow::push`
consist of two parts: a pull task and the rest to construct
a [std::span][std::span] object by which we perform
data copy in the same way as the pull task.

```cpp
hf::PushTask push1 = heteroflow.push(pull1, data1);      // copy data back to data1
hf::PullTask push2 = heteroflow.pull(pull2, data2, 10);  // copy data back to data2
```

### Kernel Task

A kernel task offloads a kernel function to a GPU device.
Heteroflow abstracts GPU memory through pull tasks 
to perform automatic device mapping and memory allocation at runtime.
Having said that, 
each pull task can convert to a pointer of whatever data type specified 
in the kernel function.

```cpp
__global__ void my_kernel1(int* data, int N);
__global__ void my_kernel2(float* data, int N);

hf::KernelTask k1 = hf.kernel(pull1, 100);  // convert the GPU data of pull1 to int*
hf::KernelTask k2 = hf.kernel(pull2, 10);   // convert the GPU data of pull2 to float*
```

The kernel task provides a rich set of methods to let you alter
the kernel configuration.

```cpp
k1.grid_x(N/256).block_x(256);              // configure the x dimension
k2.grid(N/256, 1, 1).block(256, 1, 1);      // configure the x-y-z dimension
```

Heteroflow gives users full privileges to leverage their domain-specific knowledge
to write a high-performance [CUDA][cuda-zone] kernel. 
This is an important difference between Heteroflow and existing frameworks.
Users focus on developing kernels and CPU tasks using our tasking model,
while leaving scheduling and concurrency details to Heteroflow.

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

[std::span]:             https://en.cppreference.com/w/cpp/container/span
[span::size_bytes]:      https://en.cppreference.com/w/cpp/container/span/size_bytes
[span::data]:            https://en.cppreference.com/w/cpp/container/span/data
[std::invoke]:           https://en.cppreference.com/w/cpp/utility/functional/invoke
[std::future]:           https://en.cppreference.com/w/cpp/thread/future

[span-lite]:             https://github.com/martinmoene/span-lite

[cuda-zone]:             https://developer.nvidia.com/cuda-zone
[nvcc]:                  https://developer.nvidia.com/cuda-llvm-compiler

