// Saxpy.cu
// A simple program to demonstrate single-precision A*X + Y.
#include <heteroflow/heteroflow.hpp>

__global__ void saxpy(int n, float a, float *x, float *y) {

  // Get the corresponding idx
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < n) {
    y[i] = a*x[i] + y[i];
  }
}

// Function: main
int main(void) {

  const int N = 1<<20;
  
  std::vector<float> x, y;

  hf::Executor executor(1, 1);
  hf::Heteroflow hf("simple");

  auto host_x = hf.host([&](){ x.resize(N, 1.0f); }).name("host_x");
  auto host_y = hf.host([&](){ y.resize(N, 2.0f); }).name("host_y");
  auto pull_x = hf.pull(x).name("pull_x");
  auto pull_y = hf.pull(y).name("pull_y");
  auto kernel = hf.kernel(saxpy, N, 2.0f, pull_x, pull_y)
                  .grid_x((N+255)/256)
                  .block_x(256)
                  .name("saxpy");

  auto push_x = hf.push(pull_x, x).name("push_x");
  auto push_y = hf.push(pull_y, y).name("push_y");

  host_x.precede(pull_x);
  host_y.precede(pull_y);
  kernel.precede(push_x, push_y)
        .succeed(pull_x, pull_y);

  hf.dump(std::cout);

  executor.run(hf).wait();
  
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0f));
  }
  std::cout << "Max error: " <<  maxError << '\n';

  return 0;
}
