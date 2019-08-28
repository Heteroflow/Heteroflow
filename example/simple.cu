#include <iostream>
#include <heteroflow/heteroflow.hpp>

__global__ void simple(float* X, size_t Nx, float* Y, size_t Ny) {
  printf(
    "Hello from block %d, thread %d (%lux%lu)\n", 
    blockIdx.x, threadIdx.x,
    Nx, Ny
  );
}

int main() {
  
  // create a heteroflow
  hf::Heteroflow hf("simple");

  float* h_X {nullptr};
  float* h_Y {nullptr};
  size_t n_X {100};
  size_t n_Y {200};

  auto new_X = hf.host([&](){ h_X = new float [n_X]; }).name("host_X");
  auto new_Y = hf.host([&](){ h_Y = new float [n_Y]; }).name("host_Y");
  auto gpu_X = hf.pull(h_X, n_X*sizeof(float)).name("pull_X");
  auto gpu_Y = hf.pull(h_Y, n_Y*sizeof(float)).name("pull_Y");

  // kernel task (depends on gpu_X and gpu_Y)
  auto kernel = hf.kernel(simple, gpu_X, n_X, gpu_Y, n_Y).name("kernel");

  auto push_X = hf.push(h_X, gpu_X, n_X*sizeof(float)).name("push_X");
  auto push_Y = hf.push(h_Y, gpu_Y, n_Y*sizeof(float)).name("push_Y");
  auto kill_X = hf.host([&](){ delete [] h_X; }).name("kill_X");
  auto kill_Y = hf.host([&](){ delete [] h_Y; }).name("kill_Y");

  // build up the dependency
  new_X.precede(gpu_X);
  new_Y.precede(gpu_Y);
  gpu_X.precede(kernel);
  gpu_Y.precede(kernel);
  kernel.precede(push_X, push_Y);
  push_X.precede(kill_X);
  push_Y.precede(kill_Y);
  
  // dump the heteroflow graph
  hf.dump(std::cout);
  
  // create an executor
  //hf::Executor executor;
  //executor.run(hf);

  //cudaDeviceSynchronize();

  return 0;
}





