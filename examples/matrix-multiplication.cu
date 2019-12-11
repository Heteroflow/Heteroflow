// matrix-multiplication.cu
// 
// The program implements a matrix multiplication workload 
// in two different task graphs:
//  - CPU-only tasks (baseline)
//  - CPU-GPU tasks
//
#include <heteroflow/heteroflow.hpp>

// CUDA kernel: k_multiplication
// A basic matrix multiplication kernel
__global__ void k_multiplication(
  int *a, int *b, int *c, int m, int n, int k
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  if(col < k && row < m) {
    for(int i = 0; i < n; i++) {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }
}

// Procedure: gpu
auto gpu(int M, int N, int K) {

  std::vector<int> a, b, c;

  hf::Executor executor(4, 1);  // 4 cpus 1 gpu
  hf::Heteroflow hf;

  int* ptr_a {nullptr};
  int* ptr_b {nullptr};
  int* ptr_c {nullptr};

  dim3 grid  ((K+16-1)/16, (M+16-1)/16);
  dim3 block (16, 16);

  auto ha = hf.host([&](){ 
    a.resize(M*N, M+N);
    ptr_a = a.data();
  }).name("allocate_a");

  auto hb = hf.host([&](){ 
    b.resize(N*K, N+K);
    ptr_b = b.data();
  }).name("allocate_b");

  auto hc = hf.host([&](){
    c.resize(M*K, 0);
    ptr_c = c.data();
  }).name("allocate_c");

  auto sa = hf.span(std::ref(ptr_a), M*N*sizeof(int)).name("span_a");
  auto sb = hf.span(std::ref(ptr_b), N*K*sizeof(int)).name("span_b");
  auto sc = hf.span(M*K*sizeof(int)).name("span_c");
  auto op = hf.kernel(
    grid, block, 0, k_multiplication, sa, sb, sc, M, N, K
  ).name("kernel");
  auto cc = hf.copy(std::ref(ptr_c), sc, M*K*sizeof(int)).name("push_c");
  
  ha.precede(sa);
  hb.precede(sb);
  op.succeed(sa, sb, sc).precede(cc);
  cc.succeed(hc);

  hf.dump(std::cout);

  executor.run(hf).wait();
  
  return c;  
}

// Procedure: cpu
auto cpu(int M, int N, int K) {

  std::vector<int> a, b, c;

  hf::Executor executor(4, 1);  // 4 cpus 1 gpu
  hf::Heteroflow hf;

  auto ha = hf.host([&](){ 
    a.resize(M*N, M+N);
  }).name("allocate_a");

  auto hb = hf.host([&](){ 
    b.resize(N*K, N+K);
  }).name("allocate_b");

  auto hc = hf.host([&](){
    c.resize(M*K, 0);
  }).name("allocate_c");

  auto op = hf.host([&](){
    for(int m=0; m<M; m++) {
      for(int k=0; k<K; k++) {
        for(int n=0; n<N; n++) {
          c[m*K+k] += (a[m*N+n]*b[n*K+k]);
        }
      }
    }
  }).name("c=a*b");
  
  op.succeed(ha, hb, hc);

  executor.run(hf).wait();
  
  return c;
}

int main(int argc, char* argv[]) {
  
  if(argc != 4) {
    std::cerr << "usage: matrix-multiplication M N K\n";
    std::exit(EXIT_FAILURE);
  }

  int M = std::atoi(argv[1]); 
  int N = std::atoi(argv[2]); 
  int K = std::atoi(argv[3]); 

  std::cout << "matrix A_" << M << 'x' << N << '\n'
            << "matrix B_" << N << 'x' << K << '\n'
            << "matrix C_" << M << 'x' << K << '\n';

  std::cout << "running gpu matrix multiplication ... ";
  auto gbeg = std::chrono::steady_clock::now();
  auto gres = gpu(M, N, K);
  auto gend = std::chrono::steady_clock::now();
  std::cout << "completed with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(gend-gbeg).count()
            << " ms\n";

  std::cout << "running cpu matrix multiplication ... ";
  auto cbeg = std::chrono::steady_clock::now();
  auto cres = cpu(M, N, K);
  auto cend = std::chrono::steady_clock::now();
  std::cout << "completed with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(cend-cbeg).count()
            << " ms\n";
  
  int64_t error = 0;
  std::cout << "verifying results ... ";
  for(int i=0; i<M*K; ++i) {
    error += abs(gres[i] - cres[i]);
  }
  std::cout << "abs-error=" << error << '\n';

  return 0;
}














