// Saxpy.cu
// A saxpy program to demonstrate single-precision A*X + Y.
#include <heteroflow/heteroflow.hpp>

// Kernel: saxpy
__global__ void saxpy(int n, float a, float *x, float *y) {
  // Get the corresponding idx
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a*x[i] + y[i];
  }
}

// Procedure: make_vector
template <typename T>
T* make_vector(size_t N, T value) {
  auto ptr = new T[N];
  std::fill_n(ptr, N, value);
  return ptr;
}

// Function: main
int main(void) {

  const int N = 1<<20;
  const int B = N*sizeof(float);
  
  float *x {nullptr};
  float *y {nullptr};

  hf::Executor executor(1, 1);
  hf::Heteroflow hf("saxpy");

  auto host_x = hf.host([&]{ x = make_vector(N, 1.0f); }).name("host_x");
  auto host_y = hf.host([&]{ y = make_vector(N, 2.0f); }).name("host_y"); 
  auto pull_x = hf.pull(std::ref(x), B).name("pull_x");
  auto pull_y = hf.pull(std::ref(y), B).name("pull_y");

  auto kernel = hf.kernel(saxpy, N, 2.0f, pull_x, pull_y)
                  .grid_x((N+255)/256)
                  .block_x(256)
                  .name("saxpy");

  auto push_x = hf.push(pull_x, std::ref(x), B).name("push_x");
  auto push_y = hf.push(pull_y, std::ref(y), B).name("push_y");


  host_x.precede(pull_x);
  host_y.precede(pull_y);
  kernel.precede(push_x, push_y)
        .succeed(pull_x, pull_y);
  
  // dump the graph
  hf.dump(std::cout);
  
  // run the graph
  executor.run(hf).wait();
  
  // verify the result
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0f));
  }
  std::cout << "Max error: " <<  maxError << '\n';

  return 0;
}
