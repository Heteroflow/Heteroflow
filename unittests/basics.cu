#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <heteroflow/heteroflow.hpp>

// ----------------------------------------------------------------------------
// Parameters
// ----------------------------------------------------------------------------
const size_t C = std::min(16u, std::thread::hardware_concurrency());
const size_t G = std::min(4u, hf::cuda::num_devices());

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------
template <typename T>
__global__ void k_set(T* ptr, size_t N, T value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] = value;
  }
}

template <typename T>
__global__ void k_add(T* ptr, size_t N, T value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] += value;
  }
}

template <typename T>
__global__ void k_single_add(T* ptr, size_t N, int idx, T value) {
  ptr[idx] += value;
}

// --------------------------------------------------------
// Testcase: static
// --------------------------------------------------------
TEST_CASE("static" * doctest::timeout(300)) {

  hf::Executor executor(C, G);

  REQUIRE(executor.num_cpu_workers() == C);
  REQUIRE(executor.num_gpu_workers() == G);
  REQUIRE(executor.num_workers() == C + G);

  hf::Heteroflow hf;

  REQUIRE(hf.empty() == true);
  REQUIRE(hf.size() == 0);

  hf::HostTask host;
  hf::SpanTask span;
  hf::KernelTask kernel;
  hf::CopyTask copy;

  REQUIRE(host.empty() == true);
  REQUIRE(span.empty() == true);
  REQUIRE(kernel.empty() == true);
  REQUIRE(copy.empty() == true);

  auto host2 = hf.placeholder<hf::HostTask>();
  auto span2 = hf.placeholder<hf::SpanTask>();
  auto copy2 = hf.placeholder<hf::CopyTask>();
  auto kernel2 = hf.placeholder<hf::KernelTask>();

  REQUIRE(host2.empty() == false);
  REQUIRE(span2.empty() == false);
  REQUIRE(copy2.empty() == false);
  REQUIRE(kernel2.empty() == false);
  REQUIRE(hf.size() == 4);
  REQUIRE(hf.empty() == false);

  host = host2;
  copy = copy2;
  kernel = kernel2;
  span = span2;

  REQUIRE((host == host2 && host.empty() == false));
  REQUIRE((copy == copy2 && copy.empty() == false));
  REQUIRE((span == span2 && span.empty() == false));
  REQUIRE((kernel == kernel2 && kernel.empty() == false));
  REQUIRE(hf.size() == 4);
  REQUIRE(hf.empty() == false);

  hf::HostTask host3(host2);
  hf::SpanTask span3(span2);
  hf::CopyTask copy3(copy2);
  hf::KernelTask kernel3(kernel2);

  REQUIRE((host3 == host && host2 == host));
  REQUIRE((span3 == span && span2 == span));
  REQUIRE((copy3 == copy && copy2 == copy));
  REQUIRE((kernel3 == kernel && kernel2 == kernel));
  REQUIRE(hf.size() == 4);
  REQUIRE(hf.empty() == false);
}

// --------------------------------------------------------
// Testcase: host-tasks
// --------------------------------------------------------
TEST_CASE("host-tasks" * doctest::timeout(300)) {
  
  const size_t num_tasks = 100;

  SUBCASE("Empty") {
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      REQUIRE(heteroflow.size() == 0);
      REQUIRE(heteroflow.empty() == true);
      executor.run(heteroflow).wait();
    }
  }
    
  SUBCASE("Placeholder") {
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> hosts;

      for(size_t i=0; i<num_tasks; ++i) {
        hosts.emplace_back(
          heteroflow.placeholder<hf::HostTask>().name(std::to_string(i))
        );
      }

      for(size_t i=0; i<num_tasks; ++i) {
        REQUIRE(hosts[i].name() == std::to_string(i));
        REQUIRE(hosts[i].num_dependents() == 0);
        REQUIRE(hosts[i].num_successors() == 0);
      }

      for(auto& host : hosts) {
        host.work([&counter](){ counter++; });
      }

      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks);
    }
  }

  SUBCASE("EmbarrassinglyParallel"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;

      for(size_t i=0;i<num_tasks;i++) {
        tasks.emplace_back(heteroflow.host([&counter]() {counter += 1;}));
      }

      REQUIRE(heteroflow.size() == num_tasks);
      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks);
      REQUIRE(heteroflow.size() == 100);

      counter = 0;
      
      for(size_t i=0;i<num_tasks;i++){
        tasks.emplace_back(heteroflow.host([&counter]() {counter += 1;}));
      }

      REQUIRE(heteroflow.size() == num_tasks * 2);
      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks * 2);
      REQUIRE(heteroflow.size() == 200);
    }
  }

  SUBCASE("ParallelFor") {
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);

      // Range for
      for(size_t i=0; i<num_tasks; i++) {
        hf::Heteroflow heteroflow;
        std::atomic<int> counter{0};
        auto N = ::rand() % 4098 + 1;
        std::vector<int> vec(N, 20);
        heteroflow.parallel_for(vec.begin(), vec.end(), [&](int i){
          counter += i;
        });
        executor.run(heteroflow).wait();
        auto res = std::accumulate(vec.begin(), vec.end(), 0, std::plus<int>());
        REQUIRE(counter == res);
      }

      // Index for
      for(size_t i=0; i<num_tasks; i++) {
        std::atomic<int> counter{0};
        hf::Heteroflow heteroflow;
        auto N = ::rand() % 4098 + 1;
        auto S = std::min(::rand()%10, N) + 1;
        heteroflow.parallel_for(0, N, S, [&](int){ ++counter; });
        executor.run(heteroflow).wait();
        auto res = 0;
        for(auto i=0; i<N; i+=S) {
          ++res;
        }
        REQUIRE(counter == res);
      }
    }
  }

  SUBCASE("BinarySequence"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;
      for(size_t i=0;i<num_tasks;i++){
        if(i%2 == 0){
          tasks.emplace_back(heteroflow.host(
            [&counter]() { REQUIRE(counter == 0); counter += 1;}
          ));
        }
        else{
          tasks.emplace_back(heteroflow.host(
            [&counter]() { REQUIRE(counter == 1); counter -= 1;}
          ));
        }
        if(i>0){
          tasks[i-1].precede(tasks[i]);
        }

        if(i==0) {
          REQUIRE(tasks[i].num_dependents() == 0);
        }
        else {
          REQUIRE(tasks[i].num_dependents() == 1);
        }
      }
      executor.run(heteroflow).get();
    }
  }

  SUBCASE("LinearCounter"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;
      for(size_t i=0;i<num_tasks;i++){
        tasks.emplace_back(
          heteroflow.host([&counter, i]() { 
            REQUIRE(counter == i); counter += 1;}
          )
        );
        if(i>0){
          tasks[i-1].precede(tasks[i]);
        }
      }
      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks);
      REQUIRE(heteroflow.size() == num_tasks);
    }
  }
 
  SUBCASE("Broadcast"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;
      auto src = heteroflow.host([&counter]() {counter -= 1;});
      for(size_t i=1; i<num_tasks; i++){
        auto tgt = heteroflow.host([&counter]() {REQUIRE(counter == -1);});
        src.precede(tgt);
      }
      executor.run(heteroflow).get();
      REQUIRE(counter == - 1);
      REQUIRE(heteroflow.size() == num_tasks);
    }
  }

  SUBCASE("Succeed"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;
      auto dst = heteroflow.host([&]() { REQUIRE(counter == num_tasks - 1);});
      for(size_t i=1;i<num_tasks;i++){
        auto src = heteroflow.host([&counter]() {counter += 1;});
        dst.succeed(src);
      }
      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks - 1);
      REQUIRE(heteroflow.size() == num_tasks);
    }
  }
}

// --------------------------------------------------------
// Testcase: span
// --------------------------------------------------------
TEST_CASE("span" * doctest::timeout(300)) {

  const size_t num_tasks = 4096;
    
  for(size_t c=1; c<=C; ++c) {
    for(size_t g=1; g<=G; ++g) {
      hf::Executor executor(c, g);
      hf::Heteroflow heteroflow;

      for(size_t i=0; i<num_tasks; ++i) {
        auto bytes = ::rand()% 1024;
        heteroflow.span(bytes);
      }
      
      executor.run(heteroflow).wait();
    }
  }
}

// --------------------------------------------------------
// Testcase: memset
// --------------------------------------------------------
TEST_CASE("memset" * doctest::timeout(300)) {

  const size_t num_tasks = 100;

  SUBCASE("span-fill") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata= ::rand()%4096 + 1;
          auto ptr  = new char[ndata];
          auto span = heteroflow.span(ndata);
          auto fill = heteroflow.fill(span, ndata, 'z');
          auto push = heteroflow.copy(ptr, span, ndata);
          auto host = heteroflow.host([=](){
            for(auto j=0; j<ndata; j++) {
              REQUIRE(ptr[j] == 'z');
            }
            delete [] ptr;
          });
          fill.succeed(span).precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
  
  SUBCASE("span-fill-offset") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata  = ::rand()%4096 + 1;
          auto offset = ::rand()%ndata;
          auto ptr    = new char[ndata];
          auto span   = heteroflow.span(ndata);
          auto fill1  = heteroflow.fill(span, offset, ndata-offset, 'z');
          auto fill2  = heteroflow.fill(span, offset, 'a');
          auto push   = heteroflow.copy(ptr, span, ndata);
          auto host   = heteroflow.host([=](){
            for(auto j=0; j<offset; j++) {
              REQUIRE(ptr[j] == 'a');
            }
            for(auto j=offset; j<ndata; j++) {
              REQUIRE(ptr[j] == 'z');
            }
            delete [] ptr;
          });
          fill1.succeed(span).precede(push);
          fill2.succeed(span).precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }

  SUBCASE("kernel") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata= ::rand()%4096 + 1;
          auto ptr  = new char[ndata];
          auto span = heteroflow.span(ndata);
          auto mset = heteroflow.kernel(
            (ndata+255)/256, 256, 0, k_set<char>, span, ndata, 'z'
          );
          auto push = heteroflow.copy(ptr, span, ndata);
          auto host = heteroflow.host([=](){
            for(auto j=0; j<ndata; j++) {
              REQUIRE(ptr[j] == 'z');
            }
            delete [] ptr;
          });
          span.precede(mset);
          mset.precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
  
  SUBCASE("span-fill-kernel") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata= ::rand()%4096 + 1;
          auto ptr  = new char[ndata];
          auto span = heteroflow.span(ndata);
          auto fill = heteroflow.fill(span, ndata, 'a');
          auto mset = heteroflow.kernel(
            (ndata+255)/256, 256, 0, k_add<char>, span, ndata, 1
          );
          auto push = heteroflow.copy(ptr, span, ndata);
          auto host = heteroflow.host([=](){
            for(auto j=0; j<ndata; j++) {
              REQUIRE(ptr[j] == 'b');
            }
            delete [] ptr;
          });
          span.precede(fill);
          fill.precede(mset);
          mset.precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }

  SUBCASE("pull-kernel-push") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata= ::rand()%4096 + 1;
          auto ofset= ::rand()%ndata;
          auto ptr  = new char[ndata];
          std::fill_n(ptr, ndata, 'z');
          auto span = heteroflow.span(ndata);
          auto fill = heteroflow.fill(span, ndata, 'a');
          auto mset = heteroflow.kernel(
            (ndata+255)/256, 256, 0, k_add<char>, span, ndata, 1
          );
          auto push = heteroflow.copy(ptr, span, ofset, ndata-ofset);
          auto host = heteroflow.host([=](){
            for(auto j=0; j<ndata-ofset; j++) {
              REQUIRE(ptr[j] == 'b');
            }
            for(auto j=ndata-ofset; j<ndata; j++) {
              REQUIRE(ptr[j] == 'z');
            }
            delete [] ptr;
          });
          span.precede(fill);
          fill.precede(mset);
          mset.precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
  
  SUBCASE("from-host") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata= ::rand()%4096 + 1;
          auto ptr  = new char[ndata];
          std::fill_n(ptr, ndata, 'a');
          auto span = heteroflow.span(ptr, ndata);
          auto madd = heteroflow.kernel(
            (ndata+255)/256, 256, 0, k_add<char>, span, ndata, 1
          );
          auto push = heteroflow.copy(ptr, span, ndata);
          auto host = heteroflow.host([=](){
            for(auto j=0; j<ndata; j++) {
              REQUIRE(ptr[j] == 'b');
            }
            delete [] ptr;
          });
          span.precede(madd);
          madd.precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: h2d
// --------------------------------------------------------
TEST_CASE("h2d" * doctest::timeout(300)) {

  const size_t N = 1000;
  const size_t S = 64;
  
  std::vector<std::vector<char>> res(S);
  for(auto& v : res) {
    v.resize(N);
  }

  std::vector<char> vec(N);
  for(size_t i=0; i<N; ++i) {
    vec[i] = ::rand()%40;
  }

  for(size_t c=1; c<=C; ++c) {
    for(size_t g=1; g<=G; ++g) {

      hf::Executor executor(c, g);
      hf::Heteroflow hf;
      
      for(size_t s=0; s<S; ++s) {
        std::fill_n(res[s].begin(), N, 'z');
        auto span = hf.span(vec.size());
        auto back = hf.copy(res[s].data(), span, N);
        for(size_t i=0; i<vec.size(); ++i) {
          auto copy = hf.copy(span, i, &(vec[i]), 1);
          copy.succeed(span)
              .precede(back);
        }
      }

      executor.run(hf).wait();
      
      for(size_t s=0; s<S; ++s) {
        REQUIRE(vec == res[s]);
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: d2h
// --------------------------------------------------------
TEST_CASE("d2h" * doctest::timeout(300)) {

  const size_t N = 1000;
  const size_t S = 64;
  
  std::vector<std::vector<char>> res(S);
  for(auto& v : res) {
    v.resize(N);
  }

  std::vector<char> vec(N);
  for(size_t i=0; i<N; ++i) {
    vec[i] = ::rand()%40;
  }

  for(size_t c=1; c<=C; ++c) {
    for(size_t g=1; g<=G; ++g) {

      hf::Executor executor(c, g);
      hf::Heteroflow hf;
      
      for(size_t s=0; s<S; ++s) {
        std::fill_n(res[s].begin(), N, 'z');
        auto span = hf.span(vec.data(), N);
        for(size_t i=0; i<N; ++i) {
          hf.copy(&(res[s][i]), span, i, 1)
            .succeed(span);
        }
      }

      executor.run(hf).wait();
      
      for(size_t s=0; s<S; ++s) {
        REQUIRE(vec == res[s]);
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: h2d2h
// --------------------------------------------------------
TEST_CASE("h2d2h" * doctest::timeout(300)) {

  const size_t N = 1000;
  const size_t S = 64;
  
  std::vector<std::vector<char>> res(S);
  for(auto& v : res) {
    v.resize(N);
  }

  std::vector<char> vec(N);
  for(size_t i=0; i<N; ++i) {
    vec[i] = ::rand()%40;
  }

  for(size_t c=1; c<=C; ++c) {
    for(size_t g=1; g<=G; ++g) {

      hf::Executor executor(c, g);
      hf::Heteroflow hf;
      
      for(size_t s=0; s<S; ++s) {
        std::fill_n(res[s].begin(), N, 'z');
        auto span = hf.span(vec.size());
        for(size_t i=0; i<vec.size(); ++i) {
          auto h2d = hf.copy(span, i, &(vec[i]), 1);
          auto d2h = hf.copy(&(res[s][i]), span, i, 1);
          h2d.precede(d2h).succeed(span);
        }
      }

      executor.run(hf).wait();
      
      for(size_t s=0; s<S; ++s) {
        REQUIRE(vec == res[s]);
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: d2d
// --------------------------------------------------------
TEST_CASE("d2d" * doctest::timeout(300)) {
  
  SUBCASE("without-offset") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;

        for(size_t i=0; i<100; ++i) {
          auto ndata = ::rand()%4096 + 1;
          auto data  = new char[ndata];
          auto span1 = heteroflow.span(ndata);
          auto span2 = heteroflow.span(ndata);
          auto fill1 = heteroflow.fill(span1, ndata, 'a');
          auto fill2 = heteroflow.fill(span2, ndata, 'b');
          auto kadd1 = heteroflow.kernel(
            (ndata + 255)/256, 256, 0, k_add<char>, span1, ndata, 1
          );
          auto kadd2 = heteroflow.kernel(
            (ndata + 255)/256, 256, 0, k_add<char>, span2, ndata, 1
          );
          auto trans = heteroflow.copy(
            span1, span2, ndata
          );
          auto push1 = heteroflow.copy(data, span1, ndata);
          auto test1 = heteroflow.host([data, ndata](){
            for(int i=0; i<ndata; ++i) {
              REQUIRE(data[i] == 'c');
            }
            delete [] data;
          });
          
          span1.precede(fill1);
          span2.precede(fill2);
          fill1.precede(kadd1);
          fill2.precede(kadd2);
          trans.succeed(kadd1, kadd2)
               .precede(push1);
          push1.precede(test1);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
  
  SUBCASE("with-offset") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<1024; ++i) {
          auto ndata = ::rand()%4096 + 1;
          auto offs1 = ::rand()%ndata;
          auto offs2 = ::rand()%ndata;
          auto togo  = std::min(ndata-offs1, ndata-offs2);
          auto data  = new char[ndata];
          auto span1 = heteroflow.span(ndata);
          auto span2 = heteroflow.span(ndata);
          auto fill1 = heteroflow.fill(span1, ndata, 'a');
          auto fill2 = heteroflow.fill(span2, ndata, 'b');
          auto kadd1 = heteroflow.kernel(
            (ndata + 255)/256, 256, 0, k_add<char>, span1, ndata, 1
          );
          auto kadd2 = heteroflow.kernel(
            (ndata + 255)/256, 256, 0, k_add<char>, span2, ndata, 1
          );
          auto trans = heteroflow.copy(
            span1, offs1, span2, offs2, togo
          );
          auto push1 = heteroflow.copy(data, span1, ndata);
          auto test1 = heteroflow.host([=](){
            for(int i=0; i<offs1; ++i) {
              REQUIRE(data[i] == 'b');
            }
            for(int i=offs1; i<offs1+togo; ++i) {
              REQUIRE(data[i] == 'c');
            }
            for(int i=offs1+togo; i<ndata; ++i) {
              REQUIRE(data[i] == 'b');
            }
            delete [] data;
          });
          
          span1.precede(fill1);
          span2.precede(fill2);
          fill1.precede(kadd1);
          fill2.precede(kadd2);
          trans.succeed(kadd1, kadd2)
               .precede(push1);
          push1.precede(test1);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: h2d2d2h
// --------------------------------------------------------
TEST_CASE("h2d2d2h" * doctest::timeout(300)) {

  const size_t N = 1000;
  const size_t S = 64;
  
  std::vector<std::vector<char>> res(S);
  for(auto& v : res) {
    v.resize(N);
  }

  std::vector<char> vec(N);
  for(size_t i=0; i<N; ++i) {
    vec[i] = ::rand()%40;
  }

  for(size_t c=1; c<=C; ++c) {
    for(size_t g=1; g<=G; ++g) {

      hf::Executor executor(c, g);
      hf::Heteroflow hf;
      
      for(size_t s=0; s<S; ++s) {
        std::fill_n(res[s].begin(), N, 'z');
        auto span1 = hf.span(vec.size());
        auto span2 = hf.span(vec.size());
        for(size_t i=0; i<vec.size(); ++i) {
          auto h2d = hf.copy(span1, i, &(vec[i]), 1);
          auto d2d = hf.copy(span2, i, span1, i, 1);
          auto d2h = hf.copy(&(res[s][i]), span2, i, 1);
          span1.precede(h2d);
          span2.precede(d2d);
          h2d.precede(d2d);
          d2d.precede(d2h);
        }
      }

      executor.run(hf).wait();
      
      for(size_t s=0; s<S; ++s) {
        REQUIRE(vec == res[s]);
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: dependent-copies
// --------------------------------------------------------
TEST_CASE("dependent-copies" * doctest::timeout(300)) {

  using namespace std::literals::string_literals;

  const size_t N = 1<<10;
  const size_t S = 32;

  std::vector<std::vector<char>> in(S);
  std::vector<std::vector<char>> out(S);

  std::vector<hf::CopyTask> h2d(N);
  std::vector<hf::CopyTask> d2d(N);
  std::vector<hf::CopyTask> d2h(N);
  
  // randomize the in/out data
  for(size_t s=0; s<S; ++s) {
    in[s].resize(N);
    out[s].resize(N);
    for(size_t i=0; i<N; ++i) {
      in[s][i] = ::rand()%26 + 'a';
      out[s][i] = ::rand()%26 + 'a';
    }
  }

  for(size_t c=1; c<=C; ++c) {
    for(size_t g=1; g<=G; ++g) {
      hf::Executor executor(c, g);
      hf::Heteroflow hf;
      
      for(size_t s=0; s<S; ++s) {
        auto span1 = hf.span(N).name("span1");
        auto span2 = hf.span(N).name("span2");
        
        // inter-tree dependency
        for(size_t i=1; i<N; i++) {
          h2d[i] = hf.copy(span1, i, &(in[s][i]), 1)
                     .name("h2d["s + std::to_string(i) + ']');
          d2d[i] = hf.copy(span2, i, span1, i, 1)
                     .name("d2d["s + std::to_string(i) + ']');
          d2h[i] = hf.copy(&(out[s][i]), span2, i, 1)
                     .name("d2h["s + std::to_string(i) + ']');
          h2d[i].precede(d2d[i]);
          d2d[i].precede(d2h[i]);
        }

        // tree dependency
        span1.precede(h2d[1]);
        span2.precede(h2d[1]);

        for(size_t i=1; i<N; ++i) {
          size_t l = i*2;
          size_t r = i*2 + 1;
          if(l < N) {
            h2d[i].precede(h2d[l]);
            d2d[i].precede(d2d[l]);
            d2h[i].precede(d2h[l]);
          }
          if(r < N) {
            h2d[i].precede(h2d[r]);
            d2d[i].precede(d2d[r]);
            d2h[i].precede(d2h[r]);
          }
        }
      }

      executor.run(hf).wait();
      
      for(size_t s=0; s<S; ++s) {
        for(size_t i=1; i<N; ++i) {
          REQUIRE(in[s][i] == out[s][i]);
        }
      }
    }
  }

}

// --------------------------------------------------------
// Testcase: chained-kernels
// --------------------------------------------------------
TEST_CASE("chained-kernels" * doctest::timeout(300)) {
  
  const size_t N = 1000;
  const size_t S = 64;
  const size_t L = 1000;

  std::vector<int> vec(N, 0);
  
  std::vector<std::vector<int>> res(S);
  for(auto& v : res) {
    v.resize(N);
  }

  for(size_t c=1; c<=C; ++c) {
    for(size_t g=1; g<=G; ++g) {
      hf::Executor executor(c, g);
      hf::Heteroflow hf;
      for(size_t s=0; s<S; ++s) {
        auto span = hf.span(vec.data(), N*sizeof(int));
        auto copy = hf.copy(res[s].data(), span, N*sizeof(int)).name("copy");
        hf::KernelTask prev, curr;
        for(size_t x=0; x<L; ++x) {
          curr = hf.kernel((N+16-1)/16, 16, 0, k_add<int>, span, N, 1)
                   .name(std::to_string(x) + "-kernel");
          if(x==0) {
            span.precede(curr);
          }
          else {
            prev.precede(curr);
          }
          prev = curr;
        }
        curr.precede(copy);
        auto test = hf.host([&vec=res[s]](){
          for(auto item : vec) {
            REQUIRE(item == L);
          }
        }).name("test");
        copy.precede(test);
      }
        
      executor.run(hf).wait();
    }
  }
}

// --------------------------------------------------------
// Testcase: dependent-kernels
// --------------------------------------------------------
TEST_CASE("dependent-kernels" * doctest::timeout(300)) {

  using namespace std::literals::string_literals;

  const size_t N = 1<<2;
  const size_t S = 1;

  std::vector<std::vector<char>> in(S);
  std::vector<std::vector<char>> out(S);

  std::vector<hf::CopyTask> h2d(N);
  std::vector<hf::CopyTask> d2d(N);
  std::vector<hf::CopyTask> d2h(N);
  
  // randomize the in/out data
  for(size_t s=0; s<S; ++s) {
    in[s].resize(N);
    out[s].resize(N);
    for(size_t i=0; i<N; ++i) {
      in[s][i] = ::rand()%26 + 'a';
      out[s][i] = ::rand()%26 + 'a';
    }
  }

  for(size_t c=1; c<=C; ++c) {
    for(size_t g=1; g<=G; ++g) {
      hf::Executor executor(c, g);
      hf::Heteroflow hf;
      
      for(size_t s=0; s<S; ++s) {
        auto span1 = hf.span(N).name("span1");
        auto span2 = hf.span(N).name("span2");
        
        // inter-tree dependency
        for(size_t i=1; i<N; i++) {
          h2d[i] = hf.copy(span1, i, &(in[s][i]), 1)
                     .name("h2d["s + std::to_string(i) + ']');
          d2d[i] = hf.copy(span2, i, span1, i, 1)
                     .name("d2d["s + std::to_string(i) + ']');
          d2h[i] = hf.copy(&(out[s][i]), span2, i, 1)
                     .name("d2h["s + std::to_string(i) + ']');

          auto k1 = hf.kernel(1, 1, 0, 
            k_single_add<char>, span1, N, i, 1
          ).name("k1["s + std::to_string(i) + ']');
          
          auto k2 = hf.kernel(1, 1, 0, 
            k_single_add<char>, span2, N, i, 1
          ).name("k2["s + std::to_string(i) + ']');

          h2d[i].precede(k1);
          k1.precede(d2d[i]);
          d2d[i].precede(k2);
          k2.precede(d2h[i]);
        }

        // tree dependency
        span1.precede(h2d[1]);
        span2.precede(h2d[1]);

        for(size_t i=1; i<N; ++i) {
          size_t l = i*2;
          size_t r = i*2 + 1;
          if(l < N) {
            h2d[i].precede(h2d[l]);
            d2d[i].precede(d2d[l]);
            d2h[i].precede(d2h[l]);
          }
          if(r < N) {
            h2d[i].precede(h2d[r]);
            d2d[i].precede(d2d[r]);
            d2h[i].precede(d2h[r]);
          }
        }
      }

      hf.dump(std::cout);

      return;

      executor.run(hf).wait();

      for(size_t s=0; s<S; ++s) {
        for(size_t i=1; i<N; ++i) {
          REQUIRE(in[s][i] + 2 == out[s][i]);
        }
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: state-transition
// --------------------------------------------------------
TEST_CASE("statefulness" * doctest::timeout(300)) {

  SUBCASE("linear-chain") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;

        std::vector<char> vec;
        size_t size = 0;
        char* data = nullptr;
        dim3 grid, block;

        auto host = heteroflow.host([&](){
          size = 1234567;
          vec.resize(size, 'a');
          data = vec.data();
          grid = (size+255)/256;
          block = 256;
        });
        auto span = heteroflow.span(std::ref(data), std::ref(size));
        auto kadd = heteroflow.kernel(
          std::ref(grid), std::ref(block), 0, k_add<char>, span, std::ref(size), 1
        );
        auto push = heteroflow.copy(std::ref(data), span, std::ref(size));
        auto test = heteroflow.host([&](){
          REQUIRE(size == vec.size());
          REQUIRE(data == vec.data());
          REQUIRE(grid.x == (size+255)/256);
          REQUIRE(block.x == 256);
          for(auto i : vec) {
            REQUIRE(i == 'b');
          }
        });

        host.precede(span);
        span.precede(kadd);
        kadd.precede(push);
        push.precede(test);

        executor.run(heteroflow).wait();
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: run_n
// --------------------------------------------------------
TEST_CASE("run_n" * doctest::timeout(300)) {
  
  for(size_t c=1; c<=C; ++c) {
    for(size_t g=1; g<=G; ++g) {

      std::atomic<size_t> counter{0};

      hf::Executor executor(c, g);
      hf::Heteroflow heteroflow;
      const size_t ndata = 5000;

      for(size_t n=0; n<2*G; ++n) {
        std::vector<char> vec(ndata);
        auto data = vec.data();
        
        auto host = heteroflow.host([vec=std::move(vec)]() mutable {
          for(auto& c : vec) c = 0;
        }); 
        auto span = heteroflow.span(data, ndata);
        auto kadd = heteroflow.kernel(
          (ndata + 255)/256, 256, 0, k_add<char>, span, ndata, 1
        );
        auto push = heteroflow.copy(data, span, ndata);
        auto combine = heteroflow.host([&counter, data, ndata] () {
          for(size_t i=0; i<ndata; ++i) {
            counter += data[i];
          }
        });
        
        host.precede(span);
        span.precede(kadd);
        kadd.precede(push);
        push.precede(combine);
      }
      
      auto res = 0;
      for(size_t s=0; s<25; ++s){
        auto r = ::rand() % 5;
        res += r;
        executor.run_n(heteroflow, r).wait();
        REQUIRE(counter == res*ndata*2*G);
      }
    }
  }
}


