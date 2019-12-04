#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <heteroflow/heteroflow.hpp>

// ----------------------------------------------------------------------------
// Parameters
// ----------------------------------------------------------------------------
const size_t C = std::min(4u, std::thread::hardware_concurrency());
const size_t G = std::min(4u, hf::cuda::num_devices());

// ----------------------------------------------------------------------------
// Matrix Multiplication
// ----------------------------------------------------------------------------
__global__ void k_multiplication(
  int64_t *a, int64_t *b, int64_t *c, int m, int n, int k
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t sum = 0;
  if(col < k && row < m) {
    for(int i = 0; i < n; i++) {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }
}

TEST_CASE("multiplication" * doctest::timeout(300)) {

  std::vector<int64_t> a, b, c;

  auto matmult = [&] (unsigned cpus, unsigned gpus) {
    hf::Executor executor(cpus, gpus);
    hf::Heteroflow heteroflow;
    for(unsigned B=1; B<=16; B<<=1) {
      for(int m=1; m<=256; m<<=1) {
        for(int n=1; n<=256; n<<=1) {
          for(int k=1; k<=256; k<<=1) {

            heteroflow.clear();

            int64_t* ptr_a {nullptr};
            int64_t* ptr_b {nullptr};
            int64_t* ptr_c {nullptr};

            dim3 grid  ((k+B-1)/B, (m+B-1)/B);
            dim3 block (B, B);

            auto ha = heteroflow.host([&](){ 
              a.resize(m*n);
              std::fill_n(a.begin(), m*n, m+n);
              ptr_a = a.data();
            }).name("ha");

            auto hb = heteroflow.host([&](){ 
              b.resize(n*k);
              std::fill_n(b.begin(), n*k, n+k);
              ptr_b = b.data();
            }).name("hb");

            auto hc = heteroflow.host([&](){
              c.resize(m*k);
              ptr_c = c.data();
            }).name("hc");

            auto pa = heteroflow.pull(std::ref(ptr_a), m*n*sizeof(int64_t))
                                .name("pa");
            auto pb = heteroflow.pull(std::ref(ptr_b), n*k*sizeof(int64_t))
                                .name("pb");
            auto pc = heteroflow.pull(nullptr, m*k*sizeof(int64_t), 0)
                                .name("pc");
            auto op = heteroflow.kernel(
              grid, block, 0, k_multiplication, pa, pb, pc, m, n, k
            ).name("op");
            auto cc = heteroflow.push(std::ref(ptr_c), pc, m*k*sizeof(int64_t))
                                .name("cc");
            
            ha.precede(pa);
            hb.precede(pb);
            op.succeed(pa, pb, pc).precede(cc);
            cc.succeed(hc);

            //heteroflow.dump(std::cout);

            executor.run(heteroflow).wait();

            for(const auto& x : c) {
              REQUIRE(x == (int64_t)(m+n)*(n+k)*n);
            }
          }
        }
      }
    }
  };
  
  for(unsigned cpus=1; cpus<=C; cpus++) {
    for(unsigned gpus=1; gpus<=G; gpus++) { 
      matmult(cpus, gpus);
    }
  }
}

// ----------------------------------------------------------------------------
// Matrix Transpose
// ----------------------------------------------------------------------------
__global__ void k_transpose(int *mat_in, int *mat_out, int rows, int cols) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < cols && idy < rows) {
    unsigned int pos = idy * cols + idx;
    unsigned int trans_pos = idx * rows + idy;
    mat_out[trans_pos] = mat_in[pos];
  }
}

TEST_CASE("transpose" * doctest::timeout(300)) {
  
  std::vector<int> in, out;

  auto transpose = [&] (unsigned cpus, unsigned gpus) {

    hf::Executor executor(cpus, gpus);
    hf::Heteroflow heteroflow;

    for(unsigned B=1; B<=16; B<<=1) {
      for(int m=1; m<=256; m<<=1) {
        for(int n=1; n<=256; n<<=1) {

          heteroflow.clear();

          int* ptr_in {nullptr};
          int* ptr_out {nullptr};

          dim3 grid  ((n+B-1)/B, (m+B-1)/B);
          dim3 block (B, B);

          auto hin = heteroflow.host([&](){ 
            in.resize(m*n);
            out.resize(m*n);
            for(auto& item : in) {
              item = ::rand()%100;
            }
            ptr_in = in.data();
            ptr_out = out.data();
          }).name("ha");

          auto pin  = heteroflow.pull(std::ref(ptr_in), m*n*sizeof(int))
                                .name("pin");
          auto pout = heteroflow.pull(nullptr, m*n*sizeof(int), 0)
                                .name("pout");
          auto op = heteroflow.kernel(
            grid, block, 0, k_transpose, pin, pout, m, n
          ).name("op");
          auto cc = heteroflow.push(std::ref(ptr_out), pout, m*n*sizeof(int))
                              .name("cc");
          
          hin.precede(pin);
          op.succeed(pin, pout)
            .precede(cc);

          executor.run(heteroflow).wait();

          for(int x=0; x<m; x++) {
            for(int y=0; y<n; ++y) {
              REQUIRE(in[x*n+y] == out[y*m+x]);
            }
          }
        }
      }
    }
  };
  
  for(unsigned cpus=1; cpus<=C; cpus++) {
    for(unsigned gpus=1; gpus<=G; gpus++) { 
      transpose(cpus, gpus);
    }
  }
  
}




