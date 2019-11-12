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

__global__ void kernel(
 int* a1,
 int* a2,
 int* a3,
 int* a4,
 int* a5,
 int* a6,
 int* a7,
 int* a8,
 int* a9,
 int* a10,
 int* a11
) {
}

// Function: main
int main(void) {

  /*std::vector<int> x(10, 1);

  hf::Executor executor;
  hf::Heteroflow hf;

  auto h1 = hf.host([](){});
  auto h2 = hf.host([](){});
  auto h3 = hf.host([](){});
  auto h4 = hf.host([](){});
  auto p1 = hf.pull(x);
  auto p2 = hf.pull(x);
  auto p3 = hf.pull(x);
  auto p4 = hf.pull(x);
  auto p5 = hf.pull(x);
  auto p6 = hf.pull(x);
  auto p7 = hf.pull(x);
  auto p8 = hf.pull(x);
  auto p9 = hf.pull(x);
  auto p10 = hf.pull(x);
  auto p11 = hf.pull(x);
  auto k = hf.kernel(kernel, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
  auto s = hf.push(p1, x);

  k.precede(s)
   .succeed(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);

  h1.precede(p1);
  h2.precede(p2);
  h3.precede(p3);
  h4.precede(p4);

  hf.dump(std::cout);

  executor.run(hf).wait(); */

  const int N = 1<<20;
  
  std::vector<float> x, y;
  float *x_ptr {nullptr};
  float *y_ptr {nullptr};

  hf::Executor executor(1, 1);
  hf::Heteroflow hf("simple");

  auto host_x = hf.host([&](){ x.resize(N, 1.0f); x_ptr = x.data(); }).name("host_x");
  auto host_y = hf.host([&](){ y.resize(N, 2.0f); y_ptr = y.data(); }).name("host_y");
  auto pull_x = hf.pull(std::ref(x_ptr), N*sizeof(float)).name("pull_x");
  auto pull_y = hf.pull(std::ref(y_ptr), N*sizeof(float)).name("pull_y");

  auto kernel = hf.kernel(saxpy, N, 2.0f, pull_x, pull_y)
                  .grid_x((N+255)/256)
                  .block_x(256)
                  .name("saxpy");

  auto push_x = hf.push(pull_x, std::ref(x_ptr), N*sizeof(float)).name("push_x");
  auto push_y = hf.push(pull_y, std::ref(y_ptr), N*sizeof(float)).name("push_y");


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
