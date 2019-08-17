#include <iostream>
#include <heteroflow/heteroflow.hpp>

__global__ void simple(float* X, size_t Nx, float* Y, size_t Ny) {
}

int main() {

  hf::TaskBase t1, t2, t3, t4;

  t1.precede(&t2, &t2, &t3);

  std::cout << t1.num_dependents() << " " << t1.num_successors() << std::endl;

  float* h_X = nullptr; 
  float* h_Y;

  //HF_THROW_IF(h_X == nullptr, "f---", h_X)
  HF_CHECK_CUDA(cudaErrorInitializationError, "succeFFFFFF", h_X);
  //success                            = cudaSuccess,
  //missing_configuration              = cudaErrorMissingConfiguration,
  //memory_allocation                  = cudaErrorMemoryAllocation,
  //initialization_error               = cudaErrorInitializationError,
  
  // create a heteroflow
  /*hf::Heteroflow hf;

  auto new_X = hf.host([&](){ h_X = new float [32]; });
  auto new_Y = hf.host([&](){ h_Y = new float [64]; });
  auto gpu_X = hf.pull(h_X, n_X);
  auto gpu_Y = hf.pull(h_Y, n_Y);

  // kernel task (depends on gpu_X and gpu_Y)
  auto kernel = hf.kernel(simple, gpu_X, 32, gpu_Y, 64);

  auto push_X = hf.push(h_X, gpu_X, n_X);
  auto push_Y = hf.push(h_Y, gpu_Y, n_Y);
  auto kill_X = hf.host([&](){ delete [] h_X; });
  auto kill_Y = hf.host([&](){ delete [] h_Y; });
  
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
  hf::Executor executor;
  executor.run(hf);
*/
  return 0;
}





