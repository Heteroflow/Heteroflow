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
    hf::Heteroflow hf;
    for(unsigned B=1; B<=16; B<<=1) {
      for(int m=1; m<=256; m<<=1) {
        for(int n=1; n<=256; n<<=1) {
          for(int k=1; k<=256; k<<=1) {

            hf.clear();

            int64_t* ptr_a {nullptr};
            int64_t* ptr_b {nullptr};
            int64_t* ptr_c {nullptr};

            dim3 grid  ((k+B-1)/B, (m+B-1)/B);
            dim3 block (B, B);

            auto ha = hf.host([&](){ 
              a.resize(m*n);
              std::fill_n(a.begin(), m*n, m+n);
              ptr_a = a.data();
            }).name("ha");

            auto hb = hf.host([&](){ 
              b.resize(n*k);
              std::fill_n(b.begin(), n*k, n+k);
              ptr_b = b.data();
            }).name("hb");

            auto hc = hf.host([&](){
              c.resize(m*k);
              ptr_c = c.data();
            }).name("hc");
            
            auto sa = hf.span(m*n*sizeof(int64_t)).name("sa");
            auto sb = hf.span(n*k*sizeof(int64_t)).name("sb");
            auto sc = hf.span(m*k*sizeof(int64_t)).name("sc");

            auto pa = hf.copy(sa, std::ref(ptr_a), m*n*sizeof(int64_t))
                        .name("pa");
            auto pb = hf.copy(sb, std::ref(ptr_b), n*k*sizeof(int64_t))
                        .name("pb");

            auto op = hf.kernel(
              grid, block, 0, k_multiplication, sa, sb, sc, m, n, k
            ).name("op");

            auto cc = hf.copy(std::ref(ptr_c), sc, m*k*sizeof(int64_t))
                        .name("cc");
            
            pa.succeed(ha, sa);
            pb.succeed(hb, sb);
            op.succeed(pa, pb, sc).precede(cc);
            cc.succeed(hc);

            //hf.dump(std::cout);

            executor.run(hf).wait();

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
    hf::Heteroflow hf;

    for(unsigned B=1; B<=16; B<<=1) {
      for(int m=1; m<=256; m<<=1) {
        for(int n=1; n<=256; n<<=1) {

          hf.clear();

          int* ptr_in {nullptr};
          int* ptr_out {nullptr};

          dim3 grid  ((n+B-1)/B, (m+B-1)/B);
          dim3 block (B, B);

          auto hin = hf.host([&](){ 
            in.resize(m*n);
            out.resize(m*n);
            for(auto& item : in) {
              item = ::rand()%100;
            }
            ptr_in = in.data();
            ptr_out = out.data();
          }).name("ha");

          auto sin = hf.span(m*n*sizeof(int))
                       .name("sin");
          auto pin = hf.copy(sin, std::ref(ptr_in), m*n*sizeof(int))
                       .name("pin");
          auto sout = hf.span(m*n*sizeof(int))
                        .name("pout");
                        
          auto op = hf.kernel(
            grid, block, 0, k_transpose, sin, sout, m, n
          ).name("op");

          auto cc = hf.copy(std::ref(ptr_out), sout, m*n*sizeof(int))
                      .name("cc");
          
          hin.precede(sin);
          sin.precede(pin);
          op.succeed(pin, sout)
            .precede(cc);

          executor.run(hf).wait();

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

// ----------------------------------------------------------------------------
// vector product
// ----------------------------------------------------------------------------
__global__ void k_product(int8_t *a, int8_t *b, int8_t *c, int N) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] * b[idx];
  }
}

TEST_CASE("product" * doctest::timeout(300)) {
  
  auto product = [&] (unsigned cpus, unsigned gpus) {

    hf::Executor executor(cpus, gpus);
    hf::Heteroflow hf;

    const int num_scenarios = 1024*gpus;

    std::vector<int8_t*> cs(num_scenarios); 
    
    for(int i=0; i<num_scenarios; ++i) {
      
      auto n = ::rand()%1000 + 1;
      auto b = ::rand()%32   + 1;
      int8_t va = ::rand()%10;
      int8_t vb = ::rand()%10;
      
      auto datac = hf.host([&c=cs[i], n](){
        c = new int8_t[n];
      }).name("datac");
      auto spana = hf.span(n).name("spana");
      auto spanb = hf.span(n).name("spanb");
      auto spanc = hf.span(n).name("spanc");
      auto filla = hf.fill(spana, n, va).name("filla");
      auto fillb = hf.fill(spanb, n, vb).name("fillb");
      auto kernel= hf.kernel((n+b-1)/b, b, 0,
        k_product, spana, spanb, spanc, n
      ).name("kernel");
      auto copy  = hf.copy(std::ref(cs[i]), spanc, n).name("copy");
      auto verify= hf.host([va, vb, n, &c=cs[i]](){
        for(int i=0; i<n; ++i) {
          REQUIRE(c[i] == va*vb);
        }
        delete c;
      }).name("verify");
      
      datac.precede(copy);
      spana.precede(filla);
      spanb.precede(fillb);
      kernel.succeed(filla, fillb, spanc)
            .precede(copy);
      verify.succeed(copy);
    }

    executor.run(hf).wait();
  };
  
  for(unsigned cpus=1; cpus<=C; cpus++) {
    for(unsigned gpus=1; gpus<=G; gpus++) { 
      product(cpus, gpus);
    }
  }
}









