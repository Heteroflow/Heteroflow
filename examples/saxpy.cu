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

// Function: create_vector
float* create_vector(size_t N, float value) {
  auto ptr = new float[N];
  std::fill_n(ptr, N, value);
  return ptr;
}

// Procedure: delete_vector
void delete_vector(float* ptr) {
  delete [] ptr;
}

// Procedure: verify_result
void verify_result(float* x, float* y, size_t N) {
  // verify the result
  float maxError = 0.0f;
  for (size_t i = 0; i < N; i++) {
    maxError = std::max(maxError, abs(x[i]-1.0f));
    maxError = std::max(maxError, abs(y[i]-4.0f));
  }
  std::cout << "Max error: " <<  maxError << '\n';
}

// Function: main
int main(void) {

  const size_t N = 1<<20;
  const size_t B = N*sizeof(float);
  
  float *x {nullptr};
  float *y {nullptr};

  hf::Executor executor(1, 1);
  hf::Heteroflow hf("saxpy");

  auto host_x = hf.host([&]{ x = create_vector(N, 1.0f); }).name("create_x");
  auto host_y = hf.host([&]{ y = create_vector(N, 2.0f); }).name("create_y"); 
  auto span_x = hf.span(std::ref(x), B).name("span_x");
  auto span_y = hf.span(std::ref(y), B).name("span_y");
  auto kernel = hf.kernel((N+255)/256, 256, 0, saxpy, N, 2.0f, span_x, span_y)
                  .name("saxpy");
  auto copy_x = hf.copy(std::ref(x), span_x, B).name("copy_x");
  auto copy_y = hf.copy(std::ref(y), span_y, B).name("copy_y");
  auto verify = hf.host([&]{ verify_result(x, y, N); }).name("verify");
  auto kill_x = hf.host([&]{ delete_vector(x); }).name("delete_x");
  auto kill_y = hf.host([&]{ delete_vector(y); }).name("delete_y");

  host_x.precede(span_x);
  host_y.precede(span_y);
  kernel.precede(copy_x, copy_y)
        .succeed(span_x, span_y);
  verify.precede(kill_x, kill_y)
        .succeed(copy_x, copy_y);
  
  // dump the graph
  hf.dump(std::cout);
  
  // run the graph
  executor.run(hf).wait();
  
  return 0;
}
