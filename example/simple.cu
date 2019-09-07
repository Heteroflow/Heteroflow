#include <iostream>
#include <heteroflow/heteroflow.hpp>

__global__ void simple(float* X, size_t Nx, float* Y, size_t Ny) {
  printf(
    "Hello from block %d, thread %d (%lux%lu)\n", 
    blockIdx.x, threadIdx.x,
    Nx, Ny
  );
}

__global__ void hello_kernel(int id) {
  printf(
    "Hello from block %d, thread %d (my-id=%d)\n", 
    blockIdx.x, threadIdx.x, id
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

  //auto A = hf.host([](){std::cout << "A\n";});
  //auto B = hf.host([](){std::cout << "B\n";});
  //auto C = hf.host([](){std::cout << "C\n";});
  //auto D = hf.host([](){std::cout << "D\n";});
  //
  //A.precede(B);
  //A.precede(C);
  //B.precede(D);
  //C.precede(D);

  ////auto k1 = hf.kernel(hello_kernel, 1).name("kernel1");
  //int* ptr1 = new int [100];
  //int* ptr2 = new int [100];
  //for(int i=0; i<100; ++i) {
  //  ptr1[i] = 9;
  //  ptr2[i] = 0;
  //}
  //auto p1 = hf.pull(ptr1, 100*sizeof(int)).name("pull");
  //auto p2 = hf.push(ptr2, p1, 100*sizeof(int));
  //p1.precede(p2);
  
  // create an executor
  hf::Executor executor(1, 1);
  
  executor.run(hf).wait();

  //for(int i=0; i<100; ++i) {
  //  assert(ptr2[i] == ptr1[i]);
  //}

  //cudaDeviceSynchronize();

  return 0;
}





