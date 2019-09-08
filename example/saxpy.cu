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
  
  std::vector<int> x(10, -1), y(20, 0);

  //nonstd::span<int> sp(x.data(), 10);
  auto sp = hf::make_span(x.data(), 10);

  sp.data();
  std::cout << sp.size_bytes() << std::endl;

  for(auto& i : sp) {
    std::cout << i << std::endl;
  }

  auto spx = nonstd::as_writeable_bytes(sp);

  std::cout << spx.data() << ' ' << spx.size() << std::endl;

  //hf.pull([&](){ return nonstd::span<int>(x); });


  //hf::Heteroflow hf("simple");

  //auto h_x = hf.host([&](){ x.resize(N, 1.0f);}).name("h_x");
  //auto h_y = hf.host([&](){ y.resize(N, 2.0f);}).name("h_x");

  //auto d_x = hf.pull([&]() { return {x.data(), x.size()*sizeof(float)}; });
  //auto d_y = hf.pull([&]() { return {y.data(), y.size()*sizeof(float)}; });

  //hf.pull([&](){ return hf::as_bytes(x); })
  //
  //hf.hvec();
  //hf.dvec()

  //hf.push(
  //  [&] { return {x.data(), x.size()*sizeof(float)}; },
  //  d_x
  //);


  //std::function<std::pair<void*, size_t>()> op;


  //cudaMalloc(&d_x, N*sizeof(float)); 
  //cudaMalloc(&d_y, N*sizeof(float));

  //for (int i = 0; i < N; i++) {
  //  x[i] = 1.0f;
  //  y[i] = 2.0f;
  //}

  //cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  //// Perform SAXPY on 1M elements
  //saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  //cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  //float maxError = 0.0f;
  //for (int i = 0; i < N; i++)
  //  maxError = max(maxError, abs(y[i]-4.0f));
  //printf("Max error: %f\n", maxError);

  //cudaFree(d_x);
  //cudaFree(d_y);
  //free(x);
  //free(y);
}
